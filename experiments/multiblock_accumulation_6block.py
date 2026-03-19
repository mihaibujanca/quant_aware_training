"""Cycle 033: Easiest-first multi-block accumulation up to 6 blocks.

Uses the improved geometric method (topk channel selection + lstsq output refit)
from cycles 026-027. Target blocks in easiest-first order from cycle 007:
[7, 5, 4, 1, 3, 6].

Measures full-model PPL as blocks are accumulated one by one at 4x compression.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnLayer,
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

# Easiest-first order from cycle 007 (single-block swap damage, ascending)
TARGET_BLOCKS = [7, 5, 4, 1, 3, 6]
HIDDEN_FRACTION = 0.25   # 4× compression
SELECTION_MODE = "topk"  # best method from cycles 026-027
PROBE_TRAIN_SEQ = 200
PROBE_HOLDOUT_SEQ = 50
BATCH_SIZE = 4
SEED = 42


def build_subatlas(atlas: FfnGeometryAtlas, layer_ids: list[int]) -> FfnGeometryAtlas:
    """Return an atlas containing only the specified layer IDs."""
    id_set = set(layer_ids)
    sub_layers = tuple(lg for lg in atlas.layer_geometries if lg.layer_id in id_set)
    return FfnGeometryAtlas(
        model_family=atlas.model_family,
        dataset_name=atlas.dataset_name,
        sequence_length=atlas.sequence_length,
        layer_geometries=sub_layers,
        metadata=atlas.metadata,
    )


def eval_ppl(model: torch.nn.Module, teacher, corpus, batch_size: int) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=batch_size)
    return float(metrics["perplexity"])


# ── Main ──────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)

print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
print(f"  device: {teacher.device}")

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
print(f"  probe: {corpus.probe_train.shape}, holdout: {corpus.probe_holdout.shape}")

print("Computing baseline perplexity...")
baseline_ppl = eval_ppl(teacher.model, teacher, corpus, BATCH_SIZE)
print(f"  Baseline PPL: {baseline_ppl:.2f}")

print(f"Probing geometry for target blocks: {TARGET_BLOCKS}...")
atlas = probe_ffn_geometry(
    teacher,
    corpus,
    layer_ids=tuple(TARGET_BLOCKS),
    max_tokens_per_layer=2048,
    batch_size=BATCH_SIZE,
    jacobian_samples=8,
)
print(f"  Probed {len(atlas.layer_geometries)} blocks")

print(f"Compressing all target blocks at {HIDDEN_FRACTION:.0%} hidden fraction...")
budget = CompressionBudgetSpec(
    hidden_fraction=HIDDEN_FRACTION,
    selection_mode=SELECTION_MODE,
)
full_pack = derive_ffn_compression(atlas, budget)
compressed_by_id = {layer.layer_id: layer for layer in full_pack.layers}

# ── Accumulation curve ────────────────────────────────────────────────────────
print("\nEasiest-first accumulation:")
print(f"{'k':>3} {'blocks':>20} {'PPL':>8} {'rel_pct':>8}")
print("-" * 45)

accumulation_results = []

for k in range(1, len(TARGET_BLOCKS) + 1):
    active_blocks = TARGET_BLOCKS[:k]

    # Build pack with first k blocks only
    sub_layers = tuple(compressed_by_id[bid] for bid in active_blocks)
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

    print(f"  {k:>2}  {str(active_blocks):>22}  {ppl:8.2f}  {rel:+7.2f}%")

    # Per-block compression stats
    block_stats = []
    for bid in active_blocks:
        cl = compressed_by_id[bid]
        ratio = cl.source_param_count / max(cl.compressed_param_count, 1)
        block_stats.append({"block": bid, "ratio": float(ratio)})

    accumulation_results.append({
        "k": k,
        "blocks": active_blocks,
        "ppl": ppl,
        "rel_pct": rel,
        "blocks_stats": block_stats,
    })

    del student

# ── Verdict ───────────────────────────────────────────────────────────────────
ppl_at_3 = next(r["ppl"] for r in accumulation_results if r["k"] == 3)
ppl_at_4 = next(r["ppl"] for r in accumulation_results if r["k"] == 4)
rel_at_3 = (ppl_at_3 - baseline_ppl) / baseline_ppl * 100
rel_at_4 = (ppl_at_4 - baseline_ppl) / baseline_ppl * 100

print()
if rel_at_3 > 20.0:
    verdict = "refuted"
    print(f"Verdict: refuted (k=3 PPL {rel_at_3:+.2f}% > +20% threshold)")
elif rel_at_4 <= 10.0:
    verdict = "confirmed"
    print(f"Verdict: confirmed (k=4 PPL {rel_at_4:+.2f}% ≤ +10% threshold)")
else:
    verdict = "inconclusive"
    print(f"Verdict: inconclusive (k=3 {rel_at_3:+.2f}% within 10%, k=4 {rel_at_4:+.2f}% exceeds 10%)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ks = [r["k"] for r in accumulation_results]
rels = [r["rel_pct"] for r in accumulation_results]
ax.plot(ks, rels, marker="o", color="steelblue", ms=7, lw=2)
ax.axhline(10.0, color="green", ls="--", alpha=0.7, label="+10% threshold (confirmed)")
ax.axhline(20.0, color="red", ls="--", alpha=0.7, label="+20% threshold (refuted at k=3)")
ax.axhline(0.0, color="black", ls="-", alpha=0.3)
for r in accumulation_results:
    ax.annotate(f"{r['rel_pct']:+.1f}%", (r["k"], r["rel_pct"]),
                textcoords="offset points", xytext=(4, 4), fontsize=8)
ax.set_xlabel("Number of blocks compressed (k)")
ax.set_ylabel("Relative PPL change (%)")
ax.set_title(f"Cycle 033: Easiest-First Accumulation ({HIDDEN_FRACTION:.0%} hidden per block)\n"
             f"Blocks: {TARGET_BLOCKS}  |  Verdict: {verdict.upper()}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(ks)
plt.tight_layout()

out_png = RESULTS_DIR / "cycle_033_accumulation.png"
plt.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {out_png}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "config": {
        "target_blocks": TARGET_BLOCKS,
        "hidden_fraction": HIDDEN_FRACTION,
        "selection_mode": SELECTION_MODE,
        "probe_train_seq": PROBE_TRAIN_SEQ,
        "probe_holdout_seq": PROBE_HOLDOUT_SEQ,
        "seed": SEED,
    },
    "baseline_ppl": baseline_ppl,
    "accumulation": accumulation_results,
    "verdict": verdict,
    "key_metrics": {
        "rel_pct_k3": rel_at_3,
        "rel_pct_k4": rel_at_4,
    },
}

out_json = RESULTS_DIR / "cycle_033_accumulation.json"
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {out_json}")
print(f"\n=== DONE === Verdict: {verdict}")
