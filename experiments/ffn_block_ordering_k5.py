"""Cycle 038: Block ordering optimization for 5-block stacking — GPT-2 FFN.

Part A: All C(6,5)=6 combinations of 5 blocks from the 6 easiest at 6x.
Part B: Note on insertion order (see below).

Note: The current pipeline compresses all blocks independently using the float
model's activations. Insertion order at evaluation time doesn't change the
final compressed model — it's the same set of compressed blocks regardless.
So "reverse insertion" would give identical results. The meaningful variation
is WHICH 5 blocks, not insertion order.
"""
from __future__ import annotations

import copy
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnPack,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ── Config ────────────────────────────────────────────────────────────────────
SHAKESPEARE_PATH = str(Path(__file__).parent.parent / "data" / "shakespeare.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_6_BLOCKS = [7, 5, 4, 1, 3, 6]
HIDDEN_FRACTION = 1 / 6  # 6x
PROBE_TRAIN_SEQ = 200
PROBE_HOLDOUT_SEQ = 50
BATCH_SIZE = 4
SEED = 42


def eval_ppl(model, teacher, corpus, batch_size: int) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=batch_size)
    return float(metrics["perplexity"])


# ── Main ──────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)

print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

print("Building probe corpus...")
corpus_spec = ProbeCorpusSpec(
    teacher=teacher,
    text_path=SHAKESPEARE_PATH,
    sequence_length=128,
    probe_train_sequences=PROBE_TRAIN_SEQ,
    probe_holdout_sequences=PROBE_HOLDOUT_SEQ,
    eval_val_sequences=PROBE_HOLDOUT_SEQ,
    dataset_name="proxy_shakespeare",
)
corpus = build_probe_corpus(corpus_spec)

print("Computing baseline perplexity...")
baseline_ppl = eval_ppl(teacher.model, teacher, corpus, BATCH_SIZE)
print(f"  Baseline PPL: {baseline_ppl:.2f}")

print(f"Probing geometry for all 6 blocks: {ALL_6_BLOCKS}...")
atlas = probe_ffn_geometry(
    teacher, corpus,
    layer_ids=tuple(ALL_6_BLOCKS),
    max_tokens_per_layer=2048,
    batch_size=BATCH_SIZE,
    jacobian_samples=8,
)

# Compress all 6 blocks at 6x
budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION, selection_mode="topk")
full_pack = derive_ffn_compression(atlas, budget)
compressed_by_id = {layer.layer_id: layer for layer in full_pack.layers}

# ── Part A: All C(6,5) = 6 combinations ──────────────────────────────────────
print(f"\n{'='*60}")
print("Part A: All 5-block combinations at 6x")
print(f"{'='*60}")
print(f"{'combo':>20} {'dropped':>8} {'PPL':>8} {'rel%':>8}")
print("-" * 50)

combo_results = []

for combo in combinations(ALL_6_BLOCKS, 5):
    combo_list = list(combo)
    dropped = [b for b in ALL_6_BLOCKS if b not in combo_list][0]

    sub_layers = tuple(compressed_by_id[bid] for bid in combo_list)
    sub_pack = CompressedFfnPack(
        method=full_pack.method,
        model_family=full_pack.model_family,
        dataset_name=full_pack.dataset_name,
        layers=sub_layers,
        metadata=full_pack.metadata,
    )

    student = swap_compressed_ffns(teacher, sub_pack, inplace=False)
    ppl = eval_ppl(student, teacher, corpus, BATCH_SIZE)
    rel = (ppl - baseline_ppl) / baseline_ppl * 100
    del student

    print(f"  {str(combo_list):>18}  drop {dropped:>2}  {ppl:8.2f}  {rel:+7.2f}%")

    combo_results.append({
        "blocks": combo_list,
        "dropped": dropped,
        "ppl": float(ppl),
        "rel_pct": float(rel),
    })

# Also add the k=4 reference (best 4 blocks)
k4_blocks = [7, 5, 4, 1]
sub_layers_k4 = tuple(compressed_by_id[bid] for bid in k4_blocks)
sub_pack_k4 = CompressedFfnPack(
    method=full_pack.method,
    model_family=full_pack.model_family,
    dataset_name=full_pack.dataset_name,
    layers=sub_layers_k4,
    metadata=full_pack.metadata,
)
student_k4 = swap_compressed_ffns(teacher, sub_pack_k4, inplace=False)
ppl_k4 = eval_ppl(student_k4, teacher, corpus, BATCH_SIZE)
rel_k4 = (ppl_k4 - baseline_ppl) / baseline_ppl * 100
del student_k4
print(f"\n  k=4 reference: {k4_blocks} PPL={ppl_k4:.2f} ({rel_k4:+.2f}%)")

# ── Part B: Note on insertion order ──────────────────────────────────────────
print(f"\n{'='*60}")
print("Part B: Insertion order note")
print(f"{'='*60}")
print("  The pipeline compresses each block independently using float-model activations.")
print("  Insertion order at eval time doesn't change the final model state.")
print("  The meaningful variable is WHICH 5 blocks, tested in Part A.")

# ── Verdict ───────────────────────────────────────────────────────────────────
best = min(combo_results, key=lambda r: r["rel_pct"])
worst = max(combo_results, key=lambda r: r["rel_pct"])
spread = worst["rel_pct"] - best["rel_pct"]

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Best:  drop {best['dropped']} → {best['rel_pct']:+.2f}%")
print(f"Worst: drop {worst['dropped']} → {worst['rel_pct']:+.2f}%")
print(f"Spread: {spread:.2f}pp")
print(f"k=4 reference: {rel_k4:+.2f}%")

any_under_12 = any(r["rel_pct"] < 12.0 for r in combo_results)
all_over_15 = all(r["rel_pct"] > 15.0 for r in combo_results)

if any_under_12:
    verdict = "confirmed"
    n_under = sum(1 for r in combo_results if r["rel_pct"] < 12.0)
    print(f"Verdict: CONFIRMED — {n_under} combo(s) achieve <12% PPL")
elif all_over_15:
    verdict = "refuted"
    print(f"Verdict: REFUTED — all combos exceed 15% PPL")
else:
    verdict = "inconclusive"
    n_in_range = sum(1 for r in combo_results if 12.0 <= r["rel_pct"] <= 15.0)
    print(f"Verdict: INCONCLUSIVE — {n_in_range} combo(s) in 12-15% range")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

labels = [f"drop {r['dropped']}" for r in combo_results]
rels = [r["rel_pct"] for r in combo_results]
colors = ["green" if r < 12 else "orange" if r <= 15 else "red" for r in rels]

x = np.arange(len(labels))
bars = ax.bar(x, rels, color=colors, alpha=0.8)
ax.axhline(12.0, color="green", ls="--", alpha=0.7, label="<12% (confirmed)")
ax.axhline(15.0, color="red", ls="--", alpha=0.7, label=">15% (refuted)")
ax.axhline(rel_k4, color="steelblue", ls=":", alpha=0.7, label=f"k=4 ref ({rel_k4:+.1f}%)")

for i, r in enumerate(rels):
    ax.annotate(f"{r:+.1f}%", (i, r), textcoords="offset points",
                xytext=(0, 5), ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Relative PPL change (%)")
ax.set_title(f"Cycle 038: 5-Block Combo Sweep (6x per block)\n"
             f"Best: drop {best['dropped']} ({best['rel_pct']:+.1f}%)  |  "
             f"Spread: {spread:.1f}pp  |  Verdict: {verdict.upper()}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()

out_png = RESULTS_DIR / "cycle_038_block_ordering.png"
plt.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "config": {
        "all_6_blocks": ALL_6_BLOCKS,
        "hidden_fraction": HIDDEN_FRACTION,
        "seed": SEED,
    },
    "baseline_ppl": float(baseline_ppl),
    "k4_reference": {"blocks": k4_blocks, "ppl": float(ppl_k4), "rel_pct": float(rel_k4)},
    "combo_results": combo_results,
    "best": best,
    "worst": worst,
    "spread_pp": float(spread),
    "verdict": verdict,
}

out_json = RESULTS_DIR / "cycle_038_block_ordering.json"
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {out_json}")
