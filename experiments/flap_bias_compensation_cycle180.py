"""Cycle 180 / Item 190: FLAP-style analytic bias compensation — GPT-2 FFN k=4.

Instead of lstsq reconstruction for removed channels, compute analytic bias:
  Db = W_down[:, removed] @ E[GELU(z_removed)]
and absorb as a constant bias. Kept channels use original W_down columns.

Three conditions:
  A) topk + analytic bias (no lstsq, no polish)
  B) topk + lstsq (baseline control)
  C) topk + analytic bias + 500-step polish

Verdict:
  CONFIRMED    if bias-only PPL <= 73 OR bias+polish PPL <= 68.7
  REFUTED      if bias-only > 85 AND |bias+polish - lstsq+polish| <= 1
  INCONCLUSIVE otherwise
"""
from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressedMLP,
    CompressionBudgetSpec,
    FfnGeometryAtlas,
    FfnLayerGeometry,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _apply_activation,
    _evaluate_model,
    _fit_output_projection,
    _select_hidden_units,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
)

P = lambda *a, **kw: print(*a, **kw, flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [42, 137, 7]
EVAL_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200
POLISH_STEPS = 500
POLISH_LR = 1e-4

GPT2_FFN_HIDDEN = 3072
GPT2_MODEL_DIM = 768
BLOCKS = [1, 4, 7, 11]
TOPK_HIDDEN = 512  # 3072/6

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = RESULTS_DIR / "cycle_180_flap_bias.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_mlp(model, block_id):
    return model.transformer.h[block_id].mlp


def _joint_polish(student_model, block_ids, polish_tokens, n_steps, lr, device, seed):
    """Fine-tune c_proj weights of compressed blocks on LM loss."""
    if n_steps == 0:
        return []

    student_model.requires_grad_(False)
    params = []
    for bid in block_ids:
        mlp = _find_mlp(student_model, bid)
        if hasattr(mlp, "c_proj"):
            w = mlp.c_proj.weight
            w.requires_grad_(True)
            params.append(w)
            b = getattr(mlp.c_proj, "bias", None)
            if b is not None:
                b.requires_grad_(True)
                params.append(b)

    if not params:
        return []

    optimizer = torch.optim.Adam(params, lr=lr)
    student_model.train()
    losses = []
    n_seqs = polish_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(seed)

    for step in range(n_steps):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = polish_tokens[idx].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        optimizer.zero_grad()
        raw = student_model(inputs)
        logits = raw.logits if hasattr(raw, "logits") else raw[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    student_model.eval()
    student_model.requires_grad_(False)
    return losses


def _build_analytic_bias_model(teacher, atlas, block_ids, device):
    """Build compressed model using original W_down columns + analytic bias for removed channels."""
    model = copy.deepcopy(teacher.model)
    model.eval()
    stats = {}

    budget = CompressionBudgetSpec(
        hidden_fraction=1.0 / 6,
        selection_mode="topk",
    )

    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue
        bid = layer.layer_id
        k = TOPK_HIDDEN

        selected = _select_hidden_units(layer, budget, k)
        all_indices = torch.arange(layer.hidden_dim)
        removed_mask = torch.ones(layer.hidden_dim, dtype=torch.bool)
        removed_mask[selected] = False
        removed = all_indices[removed_mask]

        source_w_in = _standard_weight(layer.handle.c_fc).float()
        source_b_in = _standard_bias(layer.handle.c_fc)
        source_w_out = _standard_weight(layer.handle.c_proj).float()
        source_b_out = _standard_bias(layer.handle.c_proj)

        # Analytic bias: mean contribution of removed channels
        # hidden_sample is post-activation (GELU(z)), shape (N, 3072)
        mean_hidden_removed = layer.hidden_sample[:, removed].float().mean(dim=0)  # (n_removed,)

        # Bias correction = W_down[:, removed] @ mean_hidden_removed
        # For GPT-2 Conv1D: W_down is stored as (3072, 768), so W_down.T[removed] gives (n_removed, 768)
        # _standard_weight returns (out_features, in_features) for Linear, (in_features, out_features) for Conv1D
        # Actually _standard_weight normalizes: for Conv1D it returns weight.T so it's (out_features, in_features)
        # i.e. shape (768, 3072) — same as nn.Linear convention
        # So source_w_out[:, removed] is (768, n_removed)
        delta_bias = source_w_out[:, removed] @ mean_hidden_removed  # (768,)

        # Build CompressedMLP
        w_out_kept = source_w_out[:, selected].contiguous()  # (768, 512)
        b_out_new = (source_b_out.float() + delta_bias) if source_b_out is not None else delta_bias

        new_mlp = CompressedMLP(
            input_dim=source_w_in.shape[1],
            hidden_dim=k,
            output_dim=w_out_kept.shape[0],
            activation_kind=layer.activation_kind,
            c_fc_bias=source_b_in is not None,
            c_proj_bias=True,
        )
        with torch.no_grad():
            new_mlp.c_fc.weight.copy_(source_w_in[selected])
            if source_b_in is not None:
                new_mlp.c_fc.bias.copy_(source_b_in[selected])
            new_mlp.c_proj.weight.copy_(w_out_kept)
            new_mlp.c_proj.bias.copy_(b_out_new)

        new_mlp.to(device)
        model.transformer.h[bid].mlp = new_mlp

        # Compute reconstruction MSE for diagnostics
        with torch.no_grad():
            hidden_kept = layer.hidden_sample[:, selected].float()
            pred = hidden_kept @ w_out_kept.t() + b_out_new
            mse = F.mse_loss(pred, layer.output_sample.float()).item()

        stats[bid] = {
            "n_channels": k,
            "n_removed": int(removed.shape[0]),
            "analytic_bias_norm": float(delta_bias.norm().item()),
            "recon_mse": round(mse, 6),
        }
        P(f"    Block {bid}: {k} kept, {removed.shape[0]} removed, "
          f"bias_norm={delta_bias.norm():.4f}, mse={mse:.6f}")

    return model, stats


def _build_lstsq_model(teacher, atlas, block_ids, device):
    """Build compressed model using standard topk + lstsq (control)."""
    model = copy.deepcopy(teacher.model)
    model.eval()
    stats = {}

    budget = CompressionBudgetSpec(
        hidden_fraction=1.0 / 6,
        selection_mode="topk",
    )

    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue
        bid = layer.layer_id
        k = TOPK_HIDDEN

        selected = _select_hidden_units(layer, budget, k)

        source_w_in = _standard_weight(layer.handle.c_fc).float()
        source_b_in = _standard_bias(layer.handle.c_fc)
        use_output_bias = _standard_bias(layer.handle.c_proj) is not None

        hidden = layer.hidden_sample[:, selected].float()
        outputs = layer.output_sample.float()
        w_out, b_out = _fit_output_projection(hidden, outputs, use_bias=use_output_bias)

        new_mlp = CompressedMLP(
            input_dim=source_w_in.shape[1],
            hidden_dim=k,
            output_dim=w_out.shape[0],
            activation_kind=layer.activation_kind,
            c_fc_bias=source_b_in is not None,
            c_proj_bias=b_out is not None,
        )
        with torch.no_grad():
            new_mlp.c_fc.weight.copy_(source_w_in[selected])
            if source_b_in is not None:
                new_mlp.c_fc.bias.copy_(source_b_in[selected])
            new_mlp.c_proj.weight.copy_(w_out)
            if b_out is not None:
                new_mlp.c_proj.bias.copy_(b_out)

        new_mlp.to(device)
        model.transformer.h[bid].mlp = new_mlp

        with torch.no_grad():
            pred = hidden @ w_out.t()
            if b_out is not None:
                pred = pred + b_out
            mse = F.mse_loss(pred, outputs).item()

        stats[bid] = {
            "n_channels": k,
            "lstsq_mse": round(mse, 6),
        }
        P(f"    Block {bid}: {k} channels, lstsq_mse={mse:.6f}")

    return model, stats


# ===========================================================================
# Main
# ===========================================================================
P("=" * 70)
P("Cycle 180 — FLAP-style bias compensation for topk pruning")
P("=" * 70)

P("\nLoading GPT-2 teacher...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()
device = teacher.device
P(f"  Teacher on device={device}")

all_results = {"seeds": {}, "config": {
    "blocks": BLOCKS,
    "topk_hidden": TOPK_HIDDEN,
    "polish_steps": POLISH_STEPS,
    "polish_lr": POLISH_LR,
    "eval_seqs": EVAL_SEQS,
    "seq_len": SEQ_LEN,
    "seeds": SEEDS,
}}

for seed in SEEDS:
    P(f"\n{'='*60}")
    P(f"Seed {seed}")
    P(f"{'='*60}")

    from aleph.lm_geometric_compression import _set_seed
    _set_seed(seed)

    # Build corpus
    corpus_spec = ProbeCorpusSpec(
        eval_val_sequences=EVAL_SEQS,
        sequence_length=SEQ_LEN,
        shuffle_seed=seed,
        teacher=teacher,
        text_path=str(OWT_PATH),
    )
    corpus = build_probe_corpus(corpus_spec)

    # Baseline PPL
    P("\n  Evaluating float baseline...")
    base_metrics = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    ppl_base = base_metrics["perplexity"]
    P(f"  Float baseline PPL: {ppl_base:.2f}")
    if not (15 <= ppl_base <= 65):
        P(f"  ABORT: baseline PPL {ppl_base:.2f} outside [15, 65]")
        sys.exit(1)

    # Probe geometry
    P("\n  Probing FFN geometry...")
    probe_spec = ProbeCorpusSpec(
        probe_train_sequences=PROBE_TRAIN_SEQS,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
        sequence_length=SEQ_LEN,
        shuffle_seed=seed + 1000,
        teacher=teacher,
        text_path=str(OWT_PATH),
    )
    probe_corpus = build_probe_corpus(probe_spec)
    atlas = probe_ffn_geometry(
        teacher,
        probe_corpus,
        layer_ids=tuple(BLOCKS),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
    )

    # --- Condition A: analytic bias only ---
    P("\n  Building condition A (analytic bias, no lstsq)...")
    model_A, stats_A = _build_analytic_bias_model(teacher, atlas, BLOCKS, device)
    P("  Evaluating condition A...")
    metrics_A = _evaluate_model(teacher, model_A, corpus, batch_size=BATCH_SIZE)
    ppl_A = metrics_A["perplexity"]
    P(f"  Condition A PPL (analytic bias): {ppl_A:.2f}")

    # --- Condition B: topk + lstsq (control) ---
    P("\n  Building condition B (topk + lstsq)...")
    model_B, stats_B = _build_lstsq_model(teacher, atlas, BLOCKS, device)
    P("  Evaluating condition B pre-polish...")
    metrics_B_pre = _evaluate_model(teacher, model_B, corpus, batch_size=BATCH_SIZE)
    ppl_B_pre = metrics_B_pre["perplexity"]
    P(f"  Condition B PPL (lstsq, no polish): {ppl_B_pre:.2f}")

    P(f"  Polishing condition B ({POLISH_STEPS} steps)...")
    losses_B = _joint_polish(model_B, BLOCKS, corpus.eval_val, POLISH_STEPS, POLISH_LR, device, seed)
    metrics_B_post = _evaluate_model(teacher, model_B, corpus, batch_size=BATCH_SIZE)
    ppl_B_post = metrics_B_post["perplexity"]
    P(f"  Condition B PPL (lstsq + polish): {ppl_B_post:.2f}")

    # --- Condition C: analytic bias + polish ---
    P("\n  Building condition C (analytic bias + polish)...")
    model_C, _ = _build_analytic_bias_model(teacher, atlas, BLOCKS, device)
    P(f"  Polishing condition C ({POLISH_STEPS} steps)...")
    losses_C = _joint_polish(model_C, BLOCKS, corpus.eval_val, POLISH_STEPS, POLISH_LR, device, seed)
    metrics_C_post = _evaluate_model(teacher, model_C, corpus, batch_size=BATCH_SIZE)
    ppl_C_post = metrics_C_post["perplexity"]
    P(f"  Condition C PPL (analytic bias + polish): {ppl_C_post:.2f}")

    all_results["seeds"][str(seed)] = {
        "ppl_base": round(ppl_base, 2),
        "ppl_A_bias_only": round(ppl_A, 2),
        "ppl_B_lstsq_pre": round(ppl_B_pre, 2),
        "ppl_B_lstsq_post": round(ppl_B_post, 2),
        "ppl_C_bias_polish": round(ppl_C_post, 2),
        "stats_A": {str(k): v for k, v in stats_A.items()},
        "stats_B": {str(k): v for k, v in stats_B.items()},
    }
    del model_A, model_B, model_C

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
P("\n" + "=" * 70)
P("AGGREGATE RESULTS")
P("=" * 70)

seeds_data = all_results["seeds"]
n = len(seeds_data)

mean_A = sum(v["ppl_A_bias_only"] for v in seeds_data.values()) / n
mean_B_pre = sum(v["ppl_B_lstsq_pre"] for v in seeds_data.values()) / n
mean_B_post = sum(v["ppl_B_lstsq_post"] for v in seeds_data.values()) / n
mean_C = sum(v["ppl_C_bias_polish"] for v in seeds_data.values()) / n

P(f"  Mean PPL A (bias only):      {mean_A:.2f}")
P(f"  Mean PPL B (lstsq pre):      {mean_B_pre:.2f}")
P(f"  Mean PPL B (lstsq+polish):   {mean_B_post:.2f}")
P(f"  Mean PPL C (bias+polish):    {mean_C:.2f}")

# Verdict
if mean_A <= 73 or mean_C <= 68.7:
    verdict = "CONFIRMED"
    reason = (f"bias-only={mean_A:.1f}<=73" if mean_A <= 73
              else f"bias+polish={mean_C:.1f}<=68.7")
elif mean_A > 85 and abs(mean_C - mean_B_post) <= 1:
    verdict = "REFUTED"
    reason = f"bias-only={mean_A:.1f}>85, bias+polish~lstsq+polish (diff={abs(mean_C-mean_B_post):.1f})"
else:
    verdict = "INCONCLUSIVE"
    reason = f"bias-only={mean_A:.1f}, bias+polish={mean_C:.1f}, lstsq+polish={mean_B_post:.1f}"

P(f"\nVERDICT: {verdict} — {reason}")

all_results["aggregate"] = {
    "mean_ppl_A_bias_only": round(mean_A, 2),
    "mean_ppl_B_lstsq_pre": round(mean_B_pre, 2),
    "mean_ppl_B_lstsq_post": round(mean_B_post, 2),
    "mean_ppl_C_bias_polish": round(mean_C, 2),
    "verdict": verdict,
    "reason": reason,
}

with open(OUT_JSON, "w") as f:
    json.dump(all_results, f, indent=2)
P(f"\nResults saved to {OUT_JSON}")
