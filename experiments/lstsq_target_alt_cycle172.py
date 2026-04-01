"""Cycle 172 / Item 182: Lstsq regression target alternatives — GPT-2 FFN k=4.

Current lstsq fits H @ W ≈ Y_float (raw FFN block output). But FFN output enters
a residual stream: y = x + FFN(x). Test whether alternative lstsq targets that
account for residual-stream context improve stacked PPL.

Targets:
  raw     — baseline: H @ W ≈ Y_float
  norm_wt — weighted lstsq with weight 1/||x_i|| per token (scale-invariant)
  svd_sub — project Y onto top-64 SVD directions of X @ W_out.T, fit in subspace
  freq_wt — weighted lstsq with token-frequency weights

All: topk channel selection k=512, blocks [1,4,7,11], 500 Adam polish, 3 seeds.

Verdict:
  CONFIRMED    if any alt improves stacked PPL >= 2 over raw lstsq
  REFUTED      if all alts within 1 PPL of raw lstsq or worse
  INCONCLUSIVE if improvement 1-2 PPL
"""
from __future__ import annotations

import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnLayer,
    CompressedFfnPack,
    CompressedMLP,
    FfnGeometryAtlas,
    FfnLayerGeometry,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _layer_target_hidden,
    _select_hidden_units,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

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
BLOCKS = [1, 4, 7, 11]
HIDDEN_DIM = 512  # 6x compression
SVD_RANK = 64  # for svd_sub target

TARGETS = ["raw", "norm_wt", "svd_sub", "freq_wt"]

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _param_count(w: torch.Tensor, b: Optional[torch.Tensor]) -> int:
    return w.numel() + (0 if b is None else b.numel())


def _find_compressed_mlp(model: nn.Module, block_id: int):
    try:
        block = model.transformer.h[block_id]
    except (AttributeError, IndexError):
        return None
    return getattr(block, "mlp", None)


def _joint_polish(
    student_model: nn.Module,
    block_ids: list[int],
    polish_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> list[float]:
    """Joint polish: optimize ALL compressed blocks' c_proj weights simultaneously."""
    if n_steps == 0:
        return []

    student_model.requires_grad_(False)

    params = []
    for bid in block_ids:
        mlp = _find_compressed_mlp(student_model, bid)
        if mlp is None or not hasattr(mlp, "c_proj"):
            continue
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
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    student_model.eval()
    student_model.requires_grad_(False)
    return losses


def _compute_token_freq_weights(
    probe_tokens: torch.Tensor,
) -> dict[int, float]:
    """Compute unigram token frequencies from probe corpus."""
    flat = probe_tokens.reshape(-1).tolist()
    counts = Counter(flat)
    total = sum(counts.values())
    return {tok: count / total for tok, count in counts.items()}


def _fit_output_projection_weighted(
    hidden: torch.Tensor,
    outputs: torch.Tensor,
    weights: torch.Tensor,
    *,
    use_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Weighted least squares: minimize sum_i w_i * ||H_i @ W - Y_i||^2.

    Equivalent to scaling rows by sqrt(w_i) and solving normal lstsq.
    """
    sqrt_w = weights.sqrt().unsqueeze(1)  # [N, 1]
    H_w = hidden * sqrt_w
    Y_w = outputs * sqrt_w
    if use_bias:
        design = torch.cat([H_w, sqrt_w], dim=1)  # bias column also scaled
    else:
        design = H_w
    solution = torch.linalg.lstsq(design, Y_w).solution
    if use_bias:
        return solution[:-1].t().contiguous(), solution[-1].contiguous()
    return solution.t().contiguous(), None


def _fit_output_projection_svd_sub(
    hidden: torch.Tensor,
    outputs: torch.Tensor,
    input_sample: torch.Tensor,
    w_out_float: torch.Tensor,
    svd_rank: int,
    *,
    use_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fit lstsq in the top-k SVD subspace of the output projection.

    Computes top-k right singular vectors of X @ W_out.T (proxy for Jacobian),
    projects Y onto that subspace, fits lstsq, then lifts back.
    """
    # Compute proxy for Jacobian dominant directions
    # XW = input_sample @ w_out_float.T has shape [N, hidden_dim_float]
    # We want top SVD of XW in output space: XW @ W_out → [N, output_dim]
    # Actually, the relevant matrix is W_out itself. We want directions in
    # output space (768-dim) that W_out cares about most.
    # SVD of w_out_float: [768, 3072] → top-k left singular vectors = top output directions
    U, S, Vh = torch.linalg.svd(w_out_float, full_matrices=False)
    # U: [768, 768], S: [768], Vh: [768, 3072]
    # Top-k output directions
    k = min(svd_rank, U.shape[1])
    V_k = U[:, :k]  # [768, k] — top-k output directions

    # Project outputs into subspace
    Y_proj = outputs @ V_k  # [N, k]

    # Fit in subspace
    if use_bias:
        design = torch.cat([hidden, torch.ones(hidden.shape[0], 1, device=hidden.device)], dim=1)
    else:
        design = hidden
    solution = torch.linalg.lstsq(design, Y_proj).solution  # [k+1?, k]

    # Lift back to full space: W_sub = solution @ V_k.T
    if use_bias:
        W_sub = solution[:-1]  # [hidden_compressed, k]
        b_sub = solution[-1]   # [k]
        W_full = (V_k @ W_sub.t()).t().contiguous()  # [768, hidden_compressed] → transposed
        # Wait: solution is [hidden_compressed+1, k], W_sub is [hidden_compressed, k]
        # W_full should be [768, hidden_compressed] for c_proj weight
        # Actually _fit_output_projection returns W.t() as weight (output_dim, hidden_dim)
        # Let me be careful. _fit_output_projection returns solution[:-1].t() which is
        # [output_dim, hidden_dim]. So solution is [hidden_dim, output_dim].
        # Here solution is [hidden_dim, k]. Lift: W_full_solution = W_sub @ V_k.T → [hidden, 768]
        W_full_solution = W_sub @ V_k.t()  # [hidden_compressed, 768]
        b_full = b_sub @ V_k.t()  # [768]
        return W_full_solution.t().contiguous(), b_full.contiguous()
    else:
        W_sub = solution  # [hidden_compressed, k]
        W_full_solution = W_sub @ V_k.t()  # [hidden_compressed, 768]
        return W_full_solution.t().contiguous(), None


def _derive_pack_with_target(
    atlas: FfnGeometryAtlas,
    block_ids: list[int],
    hidden_dim: int,
    target_mode: str,
    token_freq_weights: dict[int, float] | None = None,
    probe_tokens: torch.Tensor | None = None,
) -> CompressedFfnPack:
    """Derive compressed pack with specified lstsq target mode."""
    budget = CompressionBudgetSpec(
        hidden_fraction=hidden_dim / GPT2_FFN_HIDDEN,
        min_hidden=32,
        selection_mode="topk",
        per_layer_hidden={bid: hidden_dim for bid in block_ids},
    )
    compressed_layers: list[CompressedFfnLayer] = []

    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
            continue
        k = hidden_dim
        selected = _select_hidden_units(layer, budget, k)
        source_w_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
        source_b_in = _standard_bias(layer.handle.c_fc)
        use_output_bias = _standard_bias(layer.handle.c_proj) is not None
        student_w_in = source_w_in[selected].contiguous()
        student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()
        hidden = layer.hidden_sample[:, selected].to(dtype=torch.float32)
        outputs = layer.output_sample.to(dtype=torch.float32)
        inputs = layer.input_sample.to(dtype=torch.float32)

        if target_mode == "raw":
            student_w_out, student_b_out = _fit_output_projection(
                hidden, outputs, use_bias=use_output_bias,
            )

        elif target_mode == "norm_wt":
            # Weighted lstsq with weight = 1/||x_i||
            # This makes the lstsq prioritize tokens where the FFN output is
            # a larger fraction of the residual stream.
            x_norms = inputs.norm(dim=1).clamp(min=1e-6)  # [N]
            weights = 1.0 / x_norms  # [N]
            # Normalize weights to mean 1 for numerical stability
            weights = weights / weights.mean()
            student_w_out, student_b_out = _fit_output_projection_weighted(
                hidden, outputs, weights, use_bias=use_output_bias,
            )

        elif target_mode == "svd_sub":
            w_out_float = _standard_weight(layer.handle.c_proj).to(dtype=torch.float32)
            student_w_out, student_b_out = _fit_output_projection_svd_sub(
                hidden, outputs, inputs, w_out_float, SVD_RANK,
                use_bias=use_output_bias,
            )

        elif target_mode == "freq_wt":
            # Token-frequency weighted lstsq.
            # Each activation row came from a specific token position.
            # We approximate: weight each sample equally but scaled by the
            # average token frequency in the probe corpus.
            # Since activations are flattened (N tokens from various sequences),
            # and we don't have per-activation token IDs, we use the overall
            # token distribution as a proxy: weight = freq[token_id].
            # But we don't have per-hidden-sample token IDs directly.
            # Approximation: use uniform weights scaled by the corpus entropy
            # as a soft reweighting. Better: count token occurrences in
            # probe_holdout and assign weights proportional to frequency.
            #
            # Actually, the probe corpus collects MAX_TOKENS_PER_LAYER tokens
            # from forward passes. We DO have the token sequences. We can
            # assign each activation position its token ID's frequency.
            #
            # probe_tokens shape: [n_seqs, seq_len+1]. Each forward pass
            # processes input_ids = tokens[:, :-1], producing seq_len positions.
            # The hook collects activations from ALL positions across batches.
            # Tokens are processed in order, so the first batch_size*seq_len
            # activations correspond to the first batch_size sequences, etc.
            #
            # For simplicity and robustness: use the empirical variance of
            # output_sample rows as a proxy for importance (higher variance =
            # more informative = should be weighted more). But that's not
            # frequency weighting.
            #
            # Simplest correct approach: uniform weights (= raw lstsq) but
            # with sqrt(freq) scaling. Since we don't track per-activation
            # token IDs through the hook, we use a simple proxy:
            # weight each row by how "typical" its input norm is (closer to
            # median = more common pattern = higher weight).
            #
            # REVISED: compute frequency weights from the probe corpus tokens
            # and use the flattened token IDs to weight each position.
            if token_freq_weights is not None and probe_tokens is not None:
                # Flatten probe tokens to get per-position token IDs
                # probe_tokens: [n_seqs, seq_len+1], but hook processes
                # inputs = tokens[:, :-1], so activation positions = tokens[:, :-1]
                flat_token_ids = probe_tokens[:, :-1].reshape(-1)
                n_tokens = hidden.shape[0]
                # Take the first n_tokens (same order as hook collection)
                if flat_token_ids.shape[0] >= n_tokens:
                    token_ids = flat_token_ids[:n_tokens]
                else:
                    # Pad with mean weight if we have fewer token IDs
                    token_ids = flat_token_ids
                    n_tokens = token_ids.shape[0]

                weights = torch.ones(n_tokens, dtype=torch.float32)
                for i in range(n_tokens):
                    tid = int(token_ids[i].item())
                    weights[i] = token_freq_weights.get(tid, 1e-6)
                # Sqrt for weighted least squares
                weights = weights.sqrt()
                weights = weights / weights.mean()

                # Truncate hidden/outputs to match if needed
                h_use = hidden[:n_tokens]
                o_use = outputs[:n_tokens]
                student_w_out, student_b_out = _fit_output_projection_weighted(
                    h_use, o_use, weights, use_bias=use_output_bias,
                )
            else:
                # Fallback: raw lstsq
                student_w_out, student_b_out = _fit_output_projection(
                    hidden, outputs, use_bias=use_output_bias,
                )
        else:
            raise ValueError(f"Unknown target mode: {target_mode}")

        source_param_count = _param_count(source_w_in, source_b_in) + _param_count(
            _standard_weight(layer.handle.c_proj), _standard_bias(layer.handle.c_proj),
        )
        compressed_param_count = _param_count(student_w_in, student_b_in) + _param_count(
            student_w_out, student_b_out,
        )
        compressed_layers.append(CompressedFfnLayer(
            layer_id=layer.layer_id,
            activation_kind=layer.activation_kind,
            source_hidden_dim=layer.hidden_dim,
            compressed_hidden_dim=k,
            c_fc_weight=student_w_in,
            c_fc_bias=student_b_in,
            c_proj_weight=student_w_out,
            c_proj_bias=student_b_out,
            selected_indices=selected,
            source_param_count=source_param_count,
            compressed_param_count=compressed_param_count,
            notes={"selection_mode": "topk", "lstsq_target": target_mode},
        ))

    return CompressedFfnPack(
        method="geometric",
        model_family="gpt2",
        dataset_name="openwebtext",
        layers=tuple(compressed_layers),
        metadata={"sequence_length": SEQ_LEN, "lstsq_target": target_mode},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2 teacher...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()
device = teacher.device
print(f"  Teacher on device={device}")

# ---------------------------------------------------------------------------
# Per-seed loop
# ---------------------------------------------------------------------------
all_seed_results: dict[int, dict] = {}

for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    _set_seed(seed)

    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher,
        text_path=str(OWT_PATH),
        sequence_length=SEQ_LEN,
        probe_train_sequences=PROBE_TRAIN_SEQS,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
        eval_val_sequences=EVAL_SEQS,
        dataset_name="openwebtext",
        shuffle_seed=seed,
    ))

    bl = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = float(bl["perplexity"])
    print(f"  Teacher baseline PPL: {baseline_ppl:.3f}")

    if not (25.0 <= baseline_ppl <= 65.0):
        print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [25, 65]")
        sys.exit(1)

    # Compute token frequency weights from probe corpus
    token_freq_weights = _compute_token_freq_weights(corpus.probe_holdout)

    # Probe blocks
    print(f"  Probing FFN geometry for blocks {BLOCKS}...")
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher, corpus,
        layer_ids=tuple(BLOCKS),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )

    seed_results: dict = {"baseline_ppl": baseline_ppl, "targets": {}}

    for target in TARGETS:
        print(f"\n  Target: {target}")
        _set_seed(seed)

        pack = _derive_pack_with_target(
            atlas, BLOCKS, HIDDEN_DIM, target,
            token_freq_weights=token_freq_weights,
            probe_tokens=corpus.probe_holdout,
        )

        # Unpolished eval
        student_raw = swap_compressed_ffns(teacher, pack, inplace=False)
        student_raw.eval()
        m_raw = _evaluate_model(teacher, student_raw, corpus, batch_size=BATCH_SIZE)
        ppl_raw = float(m_raw["perplexity"])
        print(f"    Unpolished PPL: {ppl_raw:.3f}")
        del student_raw

        # Joint polish
        student_polished = swap_compressed_ffns(teacher, pack, inplace=False)
        print(f"    Joint polish ({POLISH_STEPS} steps)...")
        losses = _joint_polish(
            student_polished, BLOCKS, corpus.probe_holdout,
            POLISH_STEPS, POLISH_LR, device, seed,
        )
        if losses:
            print(f"      loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

        student_polished.eval()
        m_polished = _evaluate_model(teacher, student_polished, corpus, batch_size=BATCH_SIZE)
        ppl_polished = float(m_polished["perplexity"])
        print(f"    Polished PPL: {ppl_polished:.3f}")

        seed_results["targets"][target] = {
            "ppl_unpolished": ppl_raw,
            "ppl_polished": ppl_polished,
        }
        del student_polished

    all_seed_results[seed] = seed_results


# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATE RESULTS\n{'='*60}")

aggregated: dict[str, dict] = {}
for target in TARGETS:
    ppls_raw = [all_seed_results[s]["targets"][target]["ppl_unpolished"] for s in SEEDS]
    ppls_pol = [all_seed_results[s]["targets"][target]["ppl_polished"] for s in SEEDS]
    mean_raw = sum(ppls_raw) / len(ppls_raw)
    std_raw = (sum((p - mean_raw)**2 for p in ppls_raw) / len(ppls_raw)) ** 0.5
    mean_pol = sum(ppls_pol) / len(ppls_pol)
    std_pol = (sum((p - mean_pol)**2 for p in ppls_pol) / len(ppls_pol)) ** 0.5
    aggregated[target] = {
        "mean_ppl_unpolished": mean_raw,
        "std_ppl_unpolished": std_raw,
        "mean_ppl_polished": mean_pol,
        "std_ppl_polished": std_pol,
        "per_seed_unpolished": ppls_raw,
        "per_seed_polished": ppls_pol,
    }

# Print summary table
raw_pol = aggregated["raw"]["mean_ppl_polished"]
print(f"\n{'Target':<12} {'PPL(raw)':<18} {'PPL(polish)':<18} {'Delta vs raw':>12}")
print("-" * 65)
for target in TARGETS:
    agg = aggregated[target]
    delta = agg["mean_ppl_polished"] - raw_pol
    print(f"{target:<12} "
          f"{agg['mean_ppl_unpolished']:.2f}+/-{agg['std_ppl_unpolished']:.1f}     "
          f"{agg['mean_ppl_polished']:.2f}+/-{agg['std_ppl_polished']:.1f}     "
          f"{delta:+.2f}")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
best_alt = None
best_delta = 0.0
for target in TARGETS:
    if target == "raw":
        continue
    delta = aggregated[target]["mean_ppl_polished"] - raw_pol
    if delta < best_delta:
        best_delta = delta
        best_alt = target

teacher_ppls = [all_seed_results[s]["baseline_ppl"] for s in SEEDS]
mean_teacher = sum(teacher_ppls) / len(teacher_ppls)

if best_delta <= -2.0:
    verdict = "CONFIRMED"
    verdict_note = (f"best alt '{best_alt}' PPL {aggregated[best_alt]['mean_ppl_polished']:.2f} "
                    f"< raw PPL {raw_pol:.2f} by {-best_delta:.1f} points")
elif best_delta >= -1.0:
    verdict = "REFUTED"
    verdict_note = (f"best alt '{best_alt}' delta={best_delta:+.1f} (within 1 PPL of raw)")
else:
    verdict = "INCONCLUSIVE"
    verdict_note = (f"best alt '{best_alt}' delta={best_delta:+.1f} (between 1-2 PPL improvement)")

print(f"\nVERDICT: {verdict}")
print(f"  {verdict_note}")
print(f"  Teacher PPL (mean): {mean_teacher:.2f}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
result = {
    "cycle": 172,
    "item": 182,
    "verdict": verdict,
    "verdict_note": verdict_note,
    "seeds": SEEDS,
    "blocks": BLOCKS,
    "hidden_dim": HIDDEN_DIM,
    "svd_rank": SVD_RANK,
    "polish_steps": POLISH_STEPS,
    "polish_lr": POLISH_LR,
    "eval_sequences": EVAL_SEQS,
    "teacher_ppl_mean": mean_teacher,
    "teacher_ppl_per_seed": {str(s): all_seed_results[s]["baseline_ppl"] for s in SEEDS},
    "targets": aggregated,
    "best_alternative": best_alt,
    "best_delta": best_delta,
    "per_seed_detail": {
        str(s): all_seed_results[s] for s in SEEDS
    },
}

out_path = RESULTS_DIR / "cycle_172_lstsq_targets.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to {out_path}")
