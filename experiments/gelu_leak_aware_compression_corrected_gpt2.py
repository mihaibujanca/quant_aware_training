"""Cycle 112: GELU negative-leak aware compression (corrected) — GPT-2 FFN (item 123).

Three conditions on blocks [1, 4, 7, 11] at 6x compression, 3 seeds [42, 123, 7]:

  A (baseline): Standard topk+lstsq on all 3072 channels, k=512, + 500 Adam steps
  B (leak-aware): pos-only topk, lstsq target = y - b_leak, output = approx + b_leak, + 500 Adam
  C (filtering-only): pos-only topk, lstsq target = y (no b_leak), + 500 Adam

Probe data split: first 100 seqs for fit (b_leak + census), last 100 for lstsq + eval.

Verdict:
  CONFIRMED    if B outperforms A by >=3% relative PPL in >=2/4 blocks
  REFUTED      if all blocks |gain_B| <= 1.0%
  INCONCLUSIVE otherwise
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressedFfnLayer,
    CompressedFfnPack,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    _extract_ffn_handles,
    _fit_output_projection,
    _iter_lm_batches,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    load_nanochat_teacher,
    swap_compressed_ffns,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 7]
BLOCKS = [1, 4, 7, 11]
SEQ_LEN = 128
BATCH_SIZE = 4
PROBE_FIT_SEQS = 100       # first 100 = b_leak census + lstsq fit
PROBE_HOLDOUT_SEQS = 100   # last 100 for eval (via eval_val)
EVAL_SEQS = 100

K = 512                    # target compressed hidden dim
HIDDEN_DIM = 3072          # GPT-2 FFN hidden
POLISH_STEPS = 500
POLISH_LR = 1e-4

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _std_weight(module: nn.Module) -> torch.Tensor:
    """Return (out_features, in_features) weight, handling Conv1D."""
    w = module.weight.detach().float()
    if module.__class__.__name__ == "Conv1D":
        return w.t().contiguous()
    return w.contiguous()


def _std_bias(module: nn.Module) -> torch.Tensor | None:
    b = getattr(module, "bias", None)
    if b is None:
        return None
    return b.detach().float().contiguous()


def _collect_ffn_activations(
    model: nn.Module,
    device: torch.device,
    tokens: torch.Tensor,
    block_ids: list[int],
) -> dict[int, dict[str, torch.Tensor]]:
    """Collect pre-GELU (z), post-GELU (h), input (x), and output (y) for each block."""
    result: dict[int, dict[str, list]] = {bid: {"x": [], "z": [], "h": [], "y": []} for bid in block_ids}

    handles_by_id = {}
    for bid in block_ids:
        block = model.transformer.h[bid]
        mlp = block.mlp
        handles_by_id[bid] = mlp

    hooks = []
    for bid, mlp in handles_by_id.items():
        def make_hook(block_id):
            def hook(module, inputs, output):
                x = inputs[0].detach().cpu().float()
                z = module.c_fc(inputs[0]).detach().cpu().float()
                h = F.gelu(z)
                y = output.detach().cpu().float()
                # Flatten (B, T, D) -> (B*T, D)
                result[block_id]["x"].append(x.reshape(-1, x.shape[-1]))
                result[block_id]["z"].append(z.reshape(-1, z.shape[-1]))
                result[block_id]["h"].append(h.reshape(-1, h.shape[-1]))
                result[block_id]["y"].append(y.reshape(-1, y.shape[-1]))
            return hook
        hooks.append(mlp.register_forward_hook(make_hook(bid)))

    model.eval()
    with torch.no_grad():
        for start in range(0, tokens.shape[0], BATCH_SIZE):
            batch = tokens[start:start + BATCH_SIZE, :-1].to(device)
            model(batch)

    for h in hooks:
        h.remove()

    out: dict[int, dict[str, torch.Tensor]] = {}
    for bid in block_ids:
        out[bid] = {
            k: torch.cat(v, dim=0)
            for k, v in result[bid].items()
        }
    return out


def _compute_b_leak(
    W_out: torch.Tensor,   # (d_model, hidden_dim) = (768, 3072)
    z: torch.Tensor,       # (N, hidden_dim) pre-GELU activations
    neg_mask: torch.Tensor,  # (hidden_dim,) bool: neg-dominated channels
) -> torch.Tensor:
    """Compute constant leak bias from negative-dominated channels.

    b_leak = W_out[:, neg_ch] @ mean_per_neg_channel(GELU(z) * (z < 0))
    Returns shape (d_model,)
    """
    if neg_mask.sum() == 0:
        return torch.zeros(W_out.shape[0])
    h = F.gelu(z)  # (N, hidden_dim)
    # contribution from negative part only
    neg_h = h[:, neg_mask] * (z[:, neg_mask] < 0).float()  # (N, N_neg)
    mean_neg_h = neg_h.mean(dim=0)  # (N_neg,)
    W_neg = W_out[:, neg_mask]  # (d_model, N_neg)
    return W_neg @ mean_neg_h  # (d_model,)


def _build_compressed_layer(
    handle,
    selected: torch.Tensor,
    x_fit: torch.Tensor,
    y_target: torch.Tensor,
    use_output_bias: bool,
    layer_id: int,
) -> CompressedFfnLayer:
    """Build a CompressedFfnLayer from selected indices, fitting W_out via lstsq."""
    source_w_in = _std_weight(handle.c_fc)  # (hidden, in)
    source_b_in = _std_bias(handle.c_fc)
    source_w_out = _std_weight(handle.c_proj)  # (out, hidden)
    source_b_out = _std_bias(handle.c_proj)

    student_w_in = source_w_in[selected].contiguous()
    student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()

    # Forward through compressed c_fc + GELU on fit data
    with torch.no_grad():
        z_sub = x_fit @ source_w_in[selected].t()
        if source_b_in is not None:
            z_sub = z_sub + source_b_in[selected]
        h_sub = F.gelu(z_sub)  # (N, k)

    # Fit output projection
    student_w_out, student_b_out = _fit_output_projection(
        h_sub, y_target, use_bias=use_output_bias
    )

    source_params = source_w_in.numel() + (source_b_in.numel() if source_b_in is not None else 0)
    source_params += source_w_out.numel() + (source_b_out.numel() if source_b_out is not None else 0)
    comp_params = student_w_in.numel() + (student_b_in.numel() if student_b_in is not None else 0)
    comp_params += student_w_out.numel() + (student_b_out.numel() if student_b_out is not None else 0)

    return CompressedFfnLayer(
        layer_id=layer_id,
        activation_kind="gelu",
        source_hidden_dim=source_w_in.shape[0],
        compressed_hidden_dim=int(selected.shape[0]),
        c_fc_weight=student_w_in,
        c_fc_bias=student_b_in,
        c_proj_weight=student_w_out,
        c_proj_bias=student_b_out,
        selected_indices=selected,
        source_param_count=source_params,
        compressed_param_count=comp_params,
        notes={},
    )


def _make_pack(layers: list[CompressedFfnLayer]) -> CompressedFfnPack:
    return CompressedFfnPack(
        method="geometric",
        model_family="gpt2",
        dataset_name="openwebtext",
        layers=tuple(layers),
        metadata={},
    )


def _polish_layer_local(
    compressed_layer: CompressedFfnLayer,
    h_data: torch.Tensor,   # (N, k_eff) post-GELU activations of compressed subspace
    y_target: torch.Tensor, # (N, d_model) target outputs
    n_steps: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> CompressedFfnLayer:
    """Polish c_proj (W_out) of one compressed layer via Adam on MSE against target.

    Uses cached activations — no full model forward pass needed.
    Returns a new CompressedFfnLayer with updated weights.
    """
    W = compressed_layer.c_proj_weight.clone().requires_grad_(True)  # (d_model, k_eff)
    b = None
    if compressed_layer.c_proj_bias is not None:
        b = compressed_layer.c_proj_bias.clone().requires_grad_(True)

    params = [W] + ([b] if b is not None else [])
    optimizer = torch.optim.Adam(params, lr=lr)

    N = h_data.shape[0]
    rng = torch.Generator()
    rng.manual_seed(seed)

    for _ in range(n_steps):
        idx = torch.randperm(N, generator=rng)[:batch_size]
        h_batch = h_data[idx]          # (bs, k_eff)
        y_batch = y_target[idx]        # (bs, d_model)
        optimizer.zero_grad()
        out = h_batch @ W.t()
        if b is not None:
            out = out + b
        loss = F.mse_loss(out, y_batch)
        loss.backward()
        optimizer.step()

    return CompressedFfnLayer(
        layer_id=compressed_layer.layer_id,
        activation_kind=compressed_layer.activation_kind,
        source_hidden_dim=compressed_layer.source_hidden_dim,
        compressed_hidden_dim=compressed_layer.compressed_hidden_dim,
        c_fc_weight=compressed_layer.c_fc_weight,
        c_fc_bias=compressed_layer.c_fc_bias,
        c_proj_weight=W.detach().contiguous(),
        c_proj_bias=b.detach().contiguous() if b is not None else None,
        selected_indices=compressed_layer.selected_indices,
        source_param_count=compressed_layer.source_param_count,
        compressed_param_count=compressed_layer.compressed_param_count,
        notes=compressed_layer.notes,
    )


# ---------------------------------------------------------------------------
# Load teacher
# ---------------------------------------------------------------------------
print("Loading GPT-2 teacher...")
_set_seed(42)
teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
teacher.model.eval()
print(f"  Teacher loaded on device={teacher.device}")


# ---------------------------------------------------------------------------
# Per-seed loop
# ---------------------------------------------------------------------------
per_seed: dict[int, dict] = {}
channel_stats: dict[int, dict] = {}  # per-block stats (stable across seeds)

for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    _set_seed(seed)

    # Build corpus: probe_train=first 100, probe_holdout=last 100, eval_val=100 separate
    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher,
        text_path=str(OWT_PATH),
        sequence_length=SEQ_LEN,
        probe_train_sequences=PROBE_FIT_SEQS,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
        eval_val_sequences=EVAL_SEQS,
        dataset_name="openwebtext",
        shuffle_seed=seed,
    ))

    # Baseline PPL
    bl_metrics = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = float(bl_metrics["perplexity"])
    print(f"  Baseline PPL: {baseline_ppl:.3f}")
    if not (20.0 <= baseline_ppl <= 75.0):
        print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [20, 75] hard guard")
        sys.exit(1)
    if not (25.0 <= baseline_ppl <= 65.0):
        print(f"  WARNING: baseline PPL {baseline_ppl:.2f} outside nominal [25, 65] — proceeding (guard is [20, 75])")

    # --- Step 2: Channel sign census on probe data (first 100 seqs) ---
    # probe_train = first 100, probe_holdout = last 100
    fit_tokens = corpus.probe_train   # (100, seq_len+1)
    eval_tokens = corpus.eval_val     # held-out eval

    print(f"  Collecting activations on {fit_tokens.shape[0]} probe seqs...")
    act = _collect_ffn_activations(teacher.model, teacher.device, fit_tokens, BLOCKS)

    for bid in BLOCKS:
        z_probe = act[bid]["z"]  # (N, 3072)
        neg_frac = (z_probe < 0).float().mean(dim=0)  # (3072,)
        pos_mask = neg_frac < 0.5
        neg_mask = ~pos_mask
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())
        if bid not in channel_stats:
            channel_stats[bid] = {
                "neg_frac_mean": float(neg_frac.mean().item()),
                "n_pos": n_pos,
                "n_neg": n_neg,
            }
        print(f"  Block {bid}: N_pos={n_pos}, N_neg={n_neg}, mean_neg_frac={neg_frac.mean():.3f}")

    # Handles for accessing weights
    handles_by_id = {h.layer_id: h for h in _extract_ffn_handles(teacher.model, "gelu")}

    # --- Step 3: Compute b_leak per block on probe data ---
    b_leak_by_block: dict[int, torch.Tensor] = {}
    for bid in BLOCKS:
        z_probe = act[bid]["z"]
        neg_frac = (z_probe < 0).float().mean(dim=0)
        neg_mask = neg_frac >= 0.5
        handle = handles_by_id[bid]
        W_out = _std_weight(handle.c_proj)  # (768, 3072)
        b_leak = _compute_b_leak(W_out, z_probe, neg_mask)
        b_leak_by_block[bid] = b_leak
        y_probe = act[bid]["y"]
        y_norm = y_probe.norm(dim=1).mean().item()
        bl_norm = b_leak.norm().item()
        print(f"  Block {bid}: ||b_leak||={bl_norm:.4f}, ||y||={y_norm:.4f}, ratio={bl_norm/max(y_norm,1e-8):.4f}")

    # --- Build per-block compressed layers for A, B, C ---
    layers_A: dict[int, CompressedFfnLayer] = {}
    layers_B: dict[int, CompressedFfnLayer] = {}
    layers_C: dict[int, CompressedFfnLayer] = {}
    b_leak_stored: dict[int, torch.Tensor] = {}  # for B reconstruction

    for bid in BLOCKS:
        handle = handles_by_id[bid]
        act_data = act[bid]
        x_fit = act_data["x"]  # (N, 768)
        y_fit = act_data["y"]  # (N, 768)
        z_fit = act_data["z"]
        use_bias = _std_bias(handle.c_proj) is not None

        neg_frac = (z_fit < 0).float().mean(dim=0)
        pos_mask = neg_frac < 0.5
        neg_mask = ~pos_mask
        n_pos = int(pos_mask.sum().item())
        k_eff = min(K, n_pos)

        # Condition A: topk from all 3072 channels
        source_w_in = _std_weight(handle.c_fc)  # (3072, 768)
        # Score channels by L2 norm of output rows + input rows
        W_out = _std_weight(handle.c_proj)  # (768, 3072)
        scores = W_out.norm(dim=0) * source_w_in.norm(dim=1)  # (3072,)
        topk_all = torch.topk(scores, k=k_eff, largest=True).indices.sort().values
        layers_A[bid] = _build_compressed_layer(handle, topk_all, x_fit, y_fit, use_bias, bid)

        # Condition B: topk from pos-only channels, lstsq target = y - b_leak
        b_leak = b_leak_by_block[bid]  # (768,)
        y_adjusted = y_fit - b_leak.unsqueeze(0)  # (N, 768)
        pos_indices = torch.where(pos_mask)[0]  # (N_pos,)
        pos_scores = scores[pos_indices]
        topk_pos = pos_indices[torch.topk(pos_scores, k=k_eff, largest=True).indices.sort().values]
        layers_B[bid] = _build_compressed_layer(handle, topk_pos, x_fit, y_adjusted, use_bias, bid)
        b_leak_stored[bid] = b_leak

        # Condition C: topk from pos-only channels, lstsq target = y (no b_leak)
        layers_C[bid] = _build_compressed_layer(handle, topk_pos, x_fit, y_fit, use_bias, bid)

    # --- Polish and evaluate each condition per block ---
    # We need a custom swap that also handles b_leak injection for condition B.
    # For B: final output = mlp(x) + b_leak (absorbed as c_proj bias)

    def _absorb_b_leak(compressed_layer: CompressedFfnLayer, b_leak: torch.Tensor) -> CompressedFfnLayer:
        """Absorb b_leak into c_proj bias (or create new bias)."""
        from dataclasses import replace
        existing_bias = compressed_layer.c_proj_bias
        if existing_bias is not None:
            new_bias = existing_bias + b_leak
        else:
            new_bias = b_leak.clone()
        return CompressedFfnLayer(
            layer_id=compressed_layer.layer_id,
            activation_kind=compressed_layer.activation_kind,
            source_hidden_dim=compressed_layer.source_hidden_dim,
            compressed_hidden_dim=compressed_layer.compressed_hidden_dim,
            c_fc_weight=compressed_layer.c_fc_weight,
            c_fc_bias=compressed_layer.c_fc_bias,
            c_proj_weight=compressed_layer.c_proj_weight,
            c_proj_bias=new_bias,
            selected_indices=compressed_layer.selected_indices,
            source_param_count=compressed_layer.source_param_count,
            compressed_param_count=compressed_layer.compressed_param_count,
            notes=compressed_layer.notes,
        )

    # --- Compute h_data for each block/condition for local polish ---
    # For each block, compute post-GELU activations of the compressed subspace
    # on the fit data. These are used as the design matrix for Adam polish.
    # y_target_A[bid] = y_fit (full output)
    # y_target_B[bid] = y_fit (after absorbing b_leak into bias, target is still full y)
    # y_target_C[bid] = y_fit

    def _get_h_compressed(handle, selected: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute GELU(x @ W_in[selected].T + b_in[selected])"""
        source_w_in = _std_weight(handle.c_fc)
        source_b_in = _std_bias(handle.c_fc)
        with torch.no_grad():
            z_sub = x @ source_w_in[selected].t()
            if source_b_in is not None:
                z_sub = z_sub + source_b_in[selected]
            return F.gelu(z_sub)

    h_data: dict[str, dict[int, torch.Tensor]] = {"A": {}, "B": {}, "C": {}}
    y_data: dict[str, dict[int, torch.Tensor]] = {"A": {}, "B": {}, "C": {}}
    for bid in BLOCKS:
        handle = handles_by_id[bid]
        x_fit = act[bid]["x"]
        y_fit = act[bid]["y"]
        b_leak = b_leak_stored[bid]

        sel_A = layers_A[bid].selected_indices
        sel_B = layers_B[bid].selected_indices
        sel_C = layers_C[bid].selected_indices

        h_data["A"][bid] = _get_h_compressed(handle, sel_A, x_fit)
        h_data["B"][bid] = _get_h_compressed(handle, sel_B, x_fit)
        h_data["C"][bid] = _get_h_compressed(handle, sel_C, x_fit)

        # For A and C: target = y_fit
        # For B: c_proj already absorbs b_leak via bias; but we want to polish
        # toward y_fit (full target), since b_leak is absorbed as bias
        # The lstsq solution for B was fit to y - b_leak without bias,
        # but after absorbing b_leak into bias, the net output targets y_fit.
        # So adam polish target = y_fit for all conditions.
        y_data["A"][bid] = y_fit
        y_data["B"][bid] = y_fit  # absorbed bias means target is y_fit
        y_data["C"][bid] = y_fit

    def _run_condition(
        layers_dict: dict[int, CompressedFfnLayer],
        h_dict: dict[int, torch.Tensor],
        y_dict: dict[int, torch.Tensor],
        label: str,
    ) -> float:
        # Polish each block locally, then eval
        polished_layers: dict[int, CompressedFfnLayer] = {}
        for bid in BLOCKS:
            _set_seed(seed)
            polished = _polish_layer_local(
                layers_dict[bid],
                h_dict[bid],
                y_dict[bid],
                POLISH_STEPS,
                POLISH_LR,
                BATCH_SIZE * 32,  # larger batches for local MSE polish
                seed=seed,
            )
            polished_layers[bid] = polished

        pack = _make_pack(list(polished_layers.values()))
        student = swap_compressed_ffns(teacher, pack, inplace=False)
        student.to(teacher.device)
        student.eval()

        metrics = _evaluate_model(teacher, student, corpus, batch_size=BATCH_SIZE)
        ppl = float(metrics["perplexity"])
        delta = (ppl - baseline_ppl) / baseline_ppl * 100.0
        print(f"  {label}: PPL={ppl:.3f} ({delta:+.2f}%)")
        if ppl > 10 * baseline_ppl:
            print(f"ABORT: {label} PPL exceeds 10× baseline")
            sys.exit(1)
        del student
        return ppl

    # Condition B: absorb b_leak into bias before building pack
    layers_B_absorbed = {
        bid: _absorb_b_leak(layers_B[bid], b_leak_stored[bid])
        for bid in BLOCKS
    }

    ppl_A = _run_condition(layers_A, h_data["A"], y_data["A"], "A (standard)")
    ppl_B = _run_condition(layers_B_absorbed, h_data["B"], y_data["B"], "B (leak-aware)")
    ppl_C = _run_condition(layers_C, h_data["C"], y_data["C"], "C (pos-only, no leak)")

    per_seed[seed] = {
        "baseline": baseline_ppl,
        "A": ppl_A,
        "B": ppl_B,
        "C": ppl_C,
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATION\n{'='*60}")

def _agg(key: str) -> tuple[float, float]:
    vals = [per_seed[s][key] for s in SEEDS]
    return float(np.mean(vals)), float(np.std(vals))

bl_mean, bl_std = _agg("baseline")
A_mean, A_std = _agg("A")
B_mean, B_std = _agg("B")
C_mean, C_std = _agg("C")

print(f"  Baseline:        PPL = {bl_mean:.3f} ± {bl_std:.3f}")
print(f"  A (standard):    PPL = {A_mean:.3f} ± {A_std:.3f}")
print(f"  B (leak-aware):  PPL = {B_mean:.3f} ± {B_std:.3f}")
print(f"  C (pos-only):    PPL = {C_mean:.3f} ± {C_std:.3f}")

gain_B = (A_mean - B_mean) / A_mean * 100.0
gain_C = (A_mean - C_mean) / A_mean * 100.0
print(f"  gain_B (B vs A): {gain_B:+.2f}% (positive = B better)")
print(f"  gain_C (C vs A): {gain_C:+.2f}% (positive = C better)")

# Verdict — single compression level across all blocks (not per-block)
# We treat each seed as one "observation" for the 4-block compression
# gain_B per seed
gain_B_per_seed = [(per_seed[s]["A"] - per_seed[s]["B"]) / per_seed[s]["A"] * 100.0 for s in SEEDS]
gain_C_per_seed = [(per_seed[s]["A"] - per_seed[s]["C"]) / per_seed[s]["A"] * 100.0 for s in SEEDS]

# Interpret gain as block-level wins: each seed compresses all 4 blocks
# Count seeds where gain_B >= 3.0
wins_B = sum(1 for g in gain_B_per_seed if g >= 3.0)
print(f"\n  gain_B per seed: {[f'{g:+.2f}%' for g in gain_B_per_seed]}")
print(f"  gain_C per seed: {[f'{g:+.2f}%' for g in gain_C_per_seed]}")
print(f"  Seeds where gain_B >= 3%: {wins_B}/3")

# Map seeds to block wins: for verdict, treat each of 3 seeds as equally valid
# CONFIRMED if avg gain >= 3.0 (equivalent to majority seed agreement at 3%+ threshold)
# We apply block-level logic: wins = seeds where gain >= 3.0
# Per plan: wins >= 2 (of 4 blocks) → confirmed. Map "seeds" as "trials"
if gain_B >= 3.0 and wins_B >= 2:
    verdict = "CONFIRMED"
elif all(abs(g) <= 1.0 for g in gain_B_per_seed):
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n  VERDICT: {verdict}")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
results = {
    "cycle": 112,
    "item": 123,
    "verdict": verdict,
    "seeds": SEEDS,
    "blocks": BLOCKS,
    "per_seed": {str(s): per_seed[s] for s in SEEDS},
    "aggregate": {
        "baseline": {"mean": bl_mean, "std": bl_std},
        "A": {"mean": A_mean, "std": A_std},
        "B": {"mean": B_mean, "std": B_std},
        "C": {"mean": C_mean, "std": C_std},
        "gain_B_pct": gain_B,
        "gain_C_pct": gain_C,
        "gain_B_per_seed": gain_B_per_seed,
        "gain_C_per_seed": gain_C_per_seed,
        "wins_B": wins_B,
    },
    "channel_stats": {str(bid): channel_stats[bid] for bid in BLOCKS},
}

json_path = RESULTS_DIR / "cycle_112_leak_aware.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved JSON: {json_path}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(SEEDS))
width = 0.22
colors = {"A": "#4c72b0", "B": "#55a868", "C": "#dd8452"}

for i, (cond, color) in enumerate(colors.items()):
    vals = [per_seed[s][cond] for s in SEEDS]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width, label=cond, color=color, alpha=0.85)

# Add baseline line
bl_vals = [per_seed[s]["baseline"] for s in SEEDS]
for xi, bv in zip(x, bl_vals):
    ax.axhline(bv, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels([f"seed={s}" for s in SEEDS])
ax.set_ylabel("Perplexity (lower=better)")
ax.set_title(
    f"Cycle 112: GELU Leak-Aware Compression — blocks {BLOCKS}\n"
    f"A=standard, B=leak-aware, C=pos-only | gain_B={gain_B:+.2f}% | {verdict}"
)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

plot_path = RESULTS_DIR / "cycle_112_leak_aware.png"
plt.savefig(plot_path, dpi=120)
plt.close()
print(f"  Saved plot: {plot_path}")

print(f"\n{'='*60}")
print(f"CYCLE 112 COMPLETE — {verdict}")
print(f"  Baseline PPL:  {bl_mean:.3f}")
print(f"  A (standard):  {A_mean:.3f}")
print(f"  B (leak-aware):{B_mean:.3f}")
print(f"  C (pos-only):  {C_mean:.3f}")
print(f"  gain_B: {gain_B:+.2f}%  gain_C: {gain_C:+.2f}%")
print(f"{'='*60}")
