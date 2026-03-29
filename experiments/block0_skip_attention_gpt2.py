"""Cycle 097: Block-0-skip attention stacking — GPT-2 (item 114).

Tests whether skipping block 0 (catastrophically sensitive, +102% PPL) allows
stacking more simultaneous attention compressions.

Three conditions:
  A. 5-block skip-0: compress [1, 3, 4, 9, 10] simultaneously (rank-11 QKV)
  B. 3-block skip-0 control: compress [1, 3, 4] (lowest single-block deltas)
  C. Position probe: compress only block 0 at rank-11; measure attention output
     L2 delta at positions 0-16 vs 112-128 (tests whether block-0 sensitivity
     is position-embedding-driven)

Verdict:
  CONFIRMED   if 5-block delta <= +10%
  REFUTED     if 5-block delta >  +20%
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
N_HEADS = 12
D_HEAD = 64
D_MODEL = 768
N_BLOCKS = 12

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Viable blocks (single-block delta <= +5%) from cycle 095
VIABLE_BLOCKS = [1, 3, 4, 9, 10]          # Condition A: 5-block skip-0
CONTROL_BLOCKS = [1, 3, 4]                # Condition B: 3-block (lowest deltas)
# Cycle 095 reference: top-3 = [1, 4, 9], mean delta = +12.98%

# Position probe windows (Condition C)
POS_EARLY = (0, 16)
POS_LATE = (112, 128)


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
    W_full = _std_weight(attn.c_attn)  # (2304, 768)

    W_new = W_full.clone()

    for proj_idx in range(3):  # 0=Q, 1=K, 2=V
        base = proj_idx * D_MODEL  # 0, 768, 1536
        for h in range(N_HEADS):
            row_start = base + h * D_HEAD
            row_end = row_start + D_HEAD
            W_h = W_full[row_start:row_end, :]  # (64, 768)
            U, S, Vt = torch.linalg.svd(W_h, full_matrices=False)
            W_approx = (U[:, :rank] * S[:rank]) @ Vt[:rank, :]  # (64, 768)
            W_new[row_start:row_end, :] = W_approx

    if attn.c_attn.__class__.__name__ == "Conv1D":
        attn.c_attn.weight.data.copy_(W_new.t())
    else:
        attn.c_attn.weight.data.copy_(W_new)


def _build_corpus(teacher, seed: int):
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
# Condition C: position probe (L2 delta after block 0 attention output)
# ---------------------------------------------------------------------------
def _position_probe(model_orig: nn.Module, teacher, seed: int,
                    block_idx: int = 0) -> dict:
    """Compress block_idx at rank-11, hook attn.c_proj output.
    Returns L2 delta at early vs late positions.
    """
    device = next(model_orig.parameters()).device

    # Build a small fixed batch of tokens from OWT
    _set_seed(seed)
    corpus = _build_corpus(teacher, seed)
    # Use first batch from eval_val
    batch_tokens = corpus.eval_val[:4]  # shape (4, 128)
    if isinstance(batch_tokens, list):
        batch_tokens = torch.stack(batch_tokens)
    batch_tokens = batch_tokens.to(device)

    # Hook storage
    captured_orig = {}
    captured_comp = {}

    def make_hook(store, key):
        def hook(module, input, output):
            store[key] = output.detach().clone()
        return hook

    # --- Original model ---
    handle_orig = model_orig.transformer.h[block_idx].attn.c_proj.register_forward_hook(
        make_hook(captured_orig, "out")
    )
    with torch.no_grad():
        model_orig(batch_tokens)
    handle_orig.remove()

    # --- Compressed copy ---
    model_comp = copy.deepcopy(model_orig)
    model_comp.eval()
    _compress_attn_block_inplace(model_comp.transformer.h[block_idx], rank=RANK)

    handle_comp = model_comp.transformer.h[block_idx].attn.c_proj.register_forward_hook(
        make_hook(captured_comp, "out")
    )
    with torch.no_grad():
        model_comp(batch_tokens)
    handle_comp.remove()
    del model_comp

    # Compare at positions
    out_orig = captured_orig["out"]  # (4, 128, 768)
    out_comp = captured_comp["out"]  # (4, 128, 768)

    def window_l2(pos_start, pos_end):
        o = out_orig[:, pos_start:pos_end, :]   # (4, window, 768)
        c = out_comp[:, pos_start:pos_end, :]
        diff_norm = (o - c).norm(dim=-1).mean().item()
        ref_norm = o.norm(dim=-1).mean().item()
        return diff_norm / (ref_norm + 1e-8)

    early_ratio = window_l2(*POS_EARLY)
    late_ratio = window_l2(*POS_LATE)
    return {"early_ratio": early_ratio, "late_ratio": late_ratio}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()

# ------------------------------------------------------------------
# Baseline PPL
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

if not (25 <= mean_baseline <= 65):
    print(f"ABORT: baseline PPL {mean_baseline:.2f} outside [25, 65]")
    sys.exit(1)

# ------------------------------------------------------------------
# Condition A: 5-block skip-0 (blocks [1, 3, 4, 9, 10])
# ------------------------------------------------------------------
print(f"\n-- Condition A: 5-block skip-0, blocks={VIABLE_BLOCKS} --")
cond_a_deltas = []
for seed in SEEDS:
    _set_seed(seed)
    corpus = _build_corpus(teacher, seed)
    baseline = baseline_ppls[seed]

    model_copy = copy.deepcopy(teacher.model)
    model_copy.eval()
    for b in VIABLE_BLOCKS:
        _compress_attn_block_inplace(model_copy.transformer.h[b], rank=RANK)

    ppl = _eval_ppl(model_copy, teacher, corpus)
    if ppl > 10 * mean_baseline:
        print(f"  ABORT: seed={seed} PPL={ppl:.1f} (>10× baseline)")
        sys.exit(1)
    delta = (ppl - baseline) / baseline * 100.0
    cond_a_deltas.append(delta)
    print(f"  seed={seed}: PPL={ppl:.3f}  delta={delta:+.2f}%")
    del model_copy

cond_a_mean = float(np.mean(cond_a_deltas))
cond_a_std = float(np.std(cond_a_deltas))
print(f"  Condition A mean: {cond_a_mean:+.2f}% ± {cond_a_std:.2f}%")

# ------------------------------------------------------------------
# Condition B: 3-block skip-0 control (blocks [1, 3, 4])
# ------------------------------------------------------------------
print(f"\n-- Condition B: 3-block control, blocks={CONTROL_BLOCKS} --")
cond_b_deltas = []
for seed in SEEDS:
    _set_seed(seed)
    corpus = _build_corpus(teacher, seed)
    baseline = baseline_ppls[seed]

    model_copy = copy.deepcopy(teacher.model)
    model_copy.eval()
    for b in CONTROL_BLOCKS:
        _compress_attn_block_inplace(model_copy.transformer.h[b], rank=RANK)

    ppl = _eval_ppl(model_copy, teacher, corpus)
    delta = (ppl - baseline) / baseline * 100.0
    cond_b_deltas.append(delta)
    print(f"  seed={seed}: PPL={ppl:.3f}  delta={delta:+.2f}%")
    del model_copy

cond_b_mean = float(np.mean(cond_b_deltas))
cond_b_std = float(np.std(cond_b_deltas))
print(f"  Condition B mean: {cond_b_mean:+.2f}% ± {cond_b_std:.2f}%")

# ------------------------------------------------------------------
# Condition C: Position probe on block 0
# ------------------------------------------------------------------
print("\n-- Condition C: position probe on block 0 --")
probe_results = []
for seed in SEEDS:
    _set_seed(seed)
    r = _position_probe(teacher.model, teacher, seed, block_idx=0)
    probe_results.append(r)
    print(f"  seed={seed}: early_ratio={r['early_ratio']:.4f}  late_ratio={r['late_ratio']:.4f}")

mean_early = float(np.mean([r["early_ratio"] for r in probe_results]))
mean_late = float(np.mean([r["late_ratio"] for r in probe_results]))
ratio_late_early = mean_late / (mean_early + 1e-8)
print(f"  mean early ratio: {mean_early:.4f}")
print(f"  mean late ratio:  {mean_late:.4f}")
print(f"  late/early ratio: {ratio_late_early:.3f} (>1 = position embedding hypothesis supported)")

# ------------------------------------------------------------------
# Verdict
# ------------------------------------------------------------------
if cond_a_mean <= 10.0:
    verdict = "CONFIRMED"
elif cond_a_mean > 20.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n=== VERDICT: {verdict} ===")
print(f"  5-block delta: {cond_a_mean:+.2f}%")
print(f"  3-block control: {cond_b_mean:+.2f}%")
print(f"  Cycle 095 reference (3-block [1,4,9]): +12.98%")

# ------------------------------------------------------------------
# Save JSON
# ------------------------------------------------------------------
results = {
    "cycle": 97,
    "item": 114,
    "rank": RANK,
    "seeds": SEEDS,
    "mean_baseline_ppl": mean_baseline,
    "baseline_ppls": {str(s): v for s, v in baseline_ppls.items()},
    "condition_a_5block_skip0": {
        "blocks": VIABLE_BLOCKS,
        "deltas": cond_a_deltas,
        "mean_delta_pct": cond_a_mean,
        "std_delta_pct": cond_a_std,
    },
    "condition_b_3block_control": {
        "blocks": CONTROL_BLOCKS,
        "deltas": cond_b_deltas,
        "mean_delta_pct": cond_b_mean,
        "std_delta_pct": cond_b_std,
    },
    "condition_c_position_probe": {
        "block_compressed": 0,
        "pos_early": list(POS_EARLY),
        "pos_late": list(POS_LATE),
        "per_seed": probe_results,
        "mean_early_ratio": mean_early,
        "mean_late_ratio": mean_late,
        "late_over_early": ratio_late_early,
    },
    "cycle_095_reference_3block_mean_delta": 12.98,
    "verdict": verdict,
}

json_path = RESULTS_DIR / "cycle_097_block0_skip_attention.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ------------------------------------------------------------------
# Bar chart: baseline / 3-block / 5-block / cycle-095 reference
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

conditions = [
    "Cycle 095\n3-block [1,4,9]\n(reference)",
    "Cond B:\n3-block [1,3,4]\n(control)",
    "Cond A:\n5-block [1,3,4,9,10]\n(skip-0)",
]
means = [12.98, cond_b_mean, cond_a_mean]
stds = [0.0, cond_b_std, cond_a_std]
colors = ["gray", "steelblue", "steelblue" if cond_a_mean <= 10.0 else ("orange" if cond_a_mean <= 20.0 else "crimson")]

bars = ax.bar(conditions, means, color=colors, alpha=0.8, width=0.5)
ax.errorbar([0, 1, 2], means, yerr=stds, fmt="none", color="black",
            capsize=6, elinewidth=2, markeredgewidth=2)
ax.axhline(10.0, color="green", linestyle="--", linewidth=1.5, label="+10% (confirmed threshold)")
ax.axhline(20.0, color="red", linestyle="--", linewidth=1.5, label="+20% (refuted threshold)")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Mean PPL delta % vs baseline (3 seeds)")
ax.set_title(
    f"Block-0-Skip Attention Stacking — GPT-2\n"
    f"Verdict: {verdict} | 5-block delta: {cond_a_mean:+.2f}% | baseline PPL: {mean_baseline:.2f}"
)
ax.legend(loc="upper left")
ax.set_ylim(bottom=-2)
plt.tight_layout()

png_path = RESULTS_DIR / "cycle_097_block0_skip.png"
plt.savefig(png_path, dpi=150)
plt.close()
print(f"Saved chart: {png_path}")

print("\n=== DONE ===")
print(f"Verdict: {verdict}")
print(f"Condition A (5-block skip-0): {cond_a_mean:+.2f}% ± {cond_a_std:.2f}%")
print(f"Condition B (3-block control): {cond_b_mean:+.2f}% ± {cond_b_std:.2f}%")
print(f"Condition C (position probe late/early): {ratio_late_early:.3f}")
