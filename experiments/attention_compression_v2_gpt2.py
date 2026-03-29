"""Cycle 095: Attention compression v2 — GPT-2 (item 109).

Per-head rank-11 SVD on QKV projections ([768, 64] each, 12 heads per block).
Compression ratio: (768*11 + 11*64) / (768*64) = 9152/49152 ≈ 5.4×.
W_O is left untouched (real block compression ~3.1× when W_O included).

For each of 12 blocks:
  - Compress all 12 heads simultaneously (rank-11 QKV)
  - Eval PPL under 3 seeds, record mean PPL delta%

Verdict:
  CONFIRMED   if ≥ 6/12 blocks have mean_delta ≤ +5%
  REFUTED     if > 6/12 blocks have mean_delta > +15%
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
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    build_probe_corpus,
    load_nanochat_teacher,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 7]
EVAL_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4
RANK = 11
RANK_MILD = 16
N_HEADS = 12
D_HEAD = 64
D_MODEL = 768
N_BLOCKS = 12

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_PASS = 5.0   # % delta to count as "passes"
THRESHOLD_FAIL = 15.0  # % delta to count as "fails"


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


def _compress_attn_block_inplace(block: nn.Module, rank: int) -> None:
    """Compress all 12 heads' Q, K, V projections in one attention block.

    GPT-2 uses Conv1D for c_attn: weight shape is (768, 3*768) in Conv1D
    (i.e., transposed: (in_features=768, out_features=2304)).
    The full (out, in) = (2304, 768) matrix contains [Q, K, V] stacked along out.
    Each head h: Q_h = W_Q[h*D_HEAD:(h+1)*D_HEAD, :]  shape [D_HEAD, D_MODEL]
    """
    attn = block.attn
    # c_attn: Conv1D, weight (768, 2304) → transposed to (2304, 768)
    W_full = _std_weight(attn.c_attn)  # (2304, 768)

    W_new = W_full.clone()

    for proj_idx in range(3):  # 0=Q, 1=K, 2=V
        base = proj_idx * D_MODEL  # 0, 768, 1536
        for h in range(N_HEADS):
            row_start = base + h * D_HEAD
            row_end = row_start + D_HEAD
            W_h = W_full[row_start:row_end, :]  # (64, 768)
            U, S, Vt = torch.linalg.svd(W_h, full_matrices=False)
            # Truncate to rank
            W_approx = (U[:, :rank] * S[:rank]) @ Vt[:rank, :]  # (64, 768)
            W_new[row_start:row_end, :] = W_approx

    # Write back into Conv1D weight (stored transposed: (768, 2304))
    if attn.c_attn.__class__.__name__ == "Conv1D":
        attn.c_attn.weight.data.copy_(W_new.t())
    else:
        attn.c_attn.weight.data.copy_(W_new)


def _build_corpus(teacher, seed: int):
    """Build eval corpus using OWT, seeded."""
    spec = ProbeCorpusSpec(
        teacher=teacher,
        text_path=str(OWT_PATH),
        sequence_length=SEQ_LEN,
        probe_train_sequences=EVAL_SEQS,
        probe_holdout_sequences=EVAL_SEQS,
        eval_val_sequences=EVAL_SEQS,
        dataset_name="openwebtext",
        shuffle_seed=seed,
    )
    return build_probe_corpus(spec)


def _eval_ppl(model: nn.Module, teacher, corpus) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)
    return float(metrics["perplexity"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()

# ------------------------------------------------------------------
# Baseline PPL (per seed)
# ------------------------------------------------------------------
print("\n-- Baseline PPL --")
baseline_ppls: dict[int, float] = {}
for seed in SEEDS:
    _set_seed(seed)
    corpus = _build_corpus(teacher, seed)
    ppl = _eval_ppl(teacher.model, teacher, corpus)
    baseline_ppls[seed] = ppl
    print(f"  seed={seed}  baseline_ppl={ppl:.3f}")

mean_baseline = float(np.mean(list(baseline_ppls.values())))
print(f"  mean baseline: {mean_baseline:.3f}")

# Abort check
if not (25 <= mean_baseline <= 65):
    print(f"ABORT: baseline PPL {mean_baseline:.2f} outside [25, 65]")
    sys.exit(1)

# ------------------------------------------------------------------
# Phase 1: single-block sweeps
# ------------------------------------------------------------------
print("\n-- Phase 1: single-block sweeps (rank-11) --")
block_results: dict[int, list[float]] = {b: [] for b in range(N_BLOCKS)}

for layer_idx in range(N_BLOCKS):
    deltas = []
    for seed in SEEDS:
        _set_seed(seed)
        corpus = _build_corpus(teacher, seed)
        baseline = baseline_ppls[seed]

        # Deep-copy the model for patching
        model_copy = copy.deepcopy(teacher.model)
        model_copy.eval()

        _compress_attn_block_inplace(model_copy.transformer.h[layer_idx], rank=RANK)

        ppl = _eval_ppl(model_copy, teacher, corpus)
        delta = (ppl - baseline) / baseline * 100.0
        deltas.append(delta)

        # Abort if diverged
        if ppl > 10 * mean_baseline:
            print(f"  ABORT: block {layer_idx}, seed {seed}, PPL={ppl:.1f} (>10× baseline)")
            sys.exit(1)

        del model_copy

    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas))
    block_results[layer_idx] = deltas
    print(f"  block {layer_idx:2d}: mean_delta={mean_delta:+.2f}%  std={std_delta:.2f}%  "
          f"({' | '.join(f'{d:+.2f}%' for d in deltas)})")

mean_deltas = {b: float(np.mean(v)) for b, v in block_results.items()}
std_deltas = {b: float(np.std(v)) for b, v in block_results.items()}

# ------------------------------------------------------------------
# Verdict
# ------------------------------------------------------------------
pass_count = sum(1 for b in range(N_BLOCKS) if mean_deltas[b] <= THRESHOLD_PASS)
fail_count = sum(1 for b in range(N_BLOCKS) if mean_deltas[b] > THRESHOLD_FAIL)

if pass_count >= 6:
    verdict = "CONFIRMED"
elif fail_count > 6:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nVerdict: {verdict} (pass_count={pass_count}, fail_count={fail_count})")

# ------------------------------------------------------------------
# Phase 2: top-3 most compressible blocks, simultaneous compression
# ------------------------------------------------------------------
print("\n-- Phase 2: top-3 most compressible blocks (simultaneous, rank-11) --")
sorted_blocks = sorted(range(N_BLOCKS), key=lambda b: mean_deltas[b])
top3 = sorted_blocks[:3]
print(f"  Top-3 most compressible: {top3} (mean deltas: {[f'{mean_deltas[b]:+.2f}%' for b in top3]})")

phase2_deltas = []
for seed in SEEDS:
    _set_seed(seed)
    corpus = _build_corpus(teacher, seed)
    baseline = baseline_ppls[seed]

    model_copy = copy.deepcopy(teacher.model)
    model_copy.eval()
    for b in top3:
        _compress_attn_block_inplace(model_copy.transformer.h[b], rank=RANK)

    ppl = _eval_ppl(model_copy, teacher, corpus)
    delta = (ppl - baseline) / baseline * 100.0
    phase2_deltas.append(delta)
    print(f"  seed={seed}: PPL={ppl:.3f}  delta={delta:+.2f}%")
    del model_copy

phase2_mean = float(np.mean(phase2_deltas))
phase2_std = float(np.std(phase2_deltas))
print(f"  Phase 2 mean: {phase2_mean:+.2f}% ± {phase2_std:.2f}%")

# ------------------------------------------------------------------
# Phase 2b: top-3 with rank-16 (milder compression)
# ------------------------------------------------------------------
print(f"\n-- Phase 2b: top-3 blocks with rank-{RANK_MILD} (milder) --")
phase2b_deltas = []
for seed in SEEDS:
    _set_seed(seed)
    corpus = _build_corpus(teacher, seed)
    baseline = baseline_ppls[seed]

    model_copy = copy.deepcopy(teacher.model)
    model_copy.eval()
    for b in top3:
        _compress_attn_block_inplace(model_copy.transformer.h[b], rank=RANK_MILD)

    ppl = _eval_ppl(model_copy, teacher, corpus)
    delta = (ppl - baseline) / baseline * 100.0
    phase2b_deltas.append(delta)
    print(f"  seed={seed}: PPL={ppl:.3f}  delta={delta:+.2f}%")
    del model_copy

phase2b_mean = float(np.mean(phase2b_deltas))
phase2b_std = float(np.std(phase2b_deltas))
print(f"  Phase 2b mean: {phase2b_mean:+.2f}% ± {phase2b_std:.2f}%")

# ------------------------------------------------------------------
# Save JSON results
# ------------------------------------------------------------------
results = {
    "cycle": 95,
    "item": 109,
    "rank": RANK,
    "seeds": SEEDS,
    "mean_baseline_ppl": mean_baseline,
    "baseline_ppls": {str(s): v for s, v in baseline_ppls.items()},
    "block_mean_delta_pct": {str(b): mean_deltas[b] for b in range(N_BLOCKS)},
    "block_std_delta_pct": {str(b): std_deltas[b] for b in range(N_BLOCKS)},
    "block_all_deltas": {str(b): block_results[b] for b in range(N_BLOCKS)},
    "pass_count": pass_count,
    "fail_count": fail_count,
    "verdict": verdict,
    "top3_blocks": top3,
    "phase2_simultaneous_rank11": {
        "deltas": phase2_deltas,
        "mean_delta_pct": phase2_mean,
        "std_delta_pct": phase2_std,
    },
    "phase2b_simultaneous_rank16": {
        "deltas": phase2b_deltas,
        "mean_delta_pct": phase2b_mean,
        "std_delta_pct": phase2b_std,
    },
}

json_path = RESULTS_DIR / "cycle_095_attention_compression_v2.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ------------------------------------------------------------------
# Bar chart
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
blocks = list(range(N_BLOCKS))
means = [mean_deltas[b] for b in blocks]
stds = [std_deltas[b] for b in blocks]
colors = ["steelblue" if m <= THRESHOLD_PASS else ("orange" if m <= THRESHOLD_FAIL else "crimson")
          for m in means]
ax.bar(blocks, means, color=colors, alpha=0.8, yerr=stds, capsize=4, error_kw={"elinewidth": 1.5})
ax.axhline(THRESHOLD_PASS, color="green", linestyle="--", linewidth=1.5, label=f"+{THRESHOLD_PASS:.0f}% threshold (pass)")
ax.axhline(THRESHOLD_FAIL, color="red", linestyle="--", linewidth=1.5, label=f"+{THRESHOLD_FAIL:.0f}% threshold (fail)")
ax.axhline(0, color="black", linewidth=0.8)
for b in top3:
    ax.bar(b, means[b], color="gold", alpha=0.9, yerr=stds[b], capsize=4, error_kw={"elinewidth": 1.5})
    ax.text(b, means[b] + stds[b] + 0.5, "★", ha="center", fontsize=10, color="goldenrod")
ax.set_xticks(blocks)
ax.set_xticklabels([f"Block {b}" for b in blocks], rotation=45, ha="right")
ax.set_xlabel("GPT-2 Attention Block")
ax.set_ylabel("Mean PPL delta % (3 seeds)")
ax.set_title(f"Attention Compression v2 — rank-11 per-head QKV (5.4x)\n"
             f"Verdict: {verdict} | pass_count={pass_count}/12 | baseline_ppl={mean_baseline:.2f}")
ax.legend(loc="upper left")
ax.set_ylim(bottom=min(min(means) - 5, -5))
plt.tight_layout()
png_path = RESULTS_DIR / "cycle_095_attention_compression_v2_barchart.png"
plt.savefig(png_path, dpi=150)
plt.close()
print(f"Saved chart: {png_path}")

print("\n=== DONE ===")
print(f"Verdict: {verdict}")
print(f"Pass count (delta <= +5%): {pass_count}/12")
print(f"Fail count (delta >  +15%): {fail_count}/12")
print(f"Phase 2 (top-3 simultaneous, rank-11): {phase2_mean:+.2f}% ± {phase2_std:.2f}%")
print(f"Phase 2b (top-3 simultaneous, rank-16): {phase2b_mean:+.2f}% ± {phase2b_std:.2f}%")
