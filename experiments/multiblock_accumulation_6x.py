"""Cycle 036: Easiest-first multi-block accumulation at 6x compression.

Identical to cycle 033 experiment except HIDDEN_FRACTION = 1/6 (6x) instead of 0.25 (4x).
Tests whether per-block 6x tolerances (confirmed in cycle 035) compose safely at k=4+.
Overlays cycle 033 uniform-4x curve for comparison.
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

TARGET_BLOCKS = [7, 5, 4, 1, 3, 6]
HIDDEN_FRACTION = 1 / 6  # 6× compression (was 0.25 = 4× in cycle 033)
SELECTION_MODE = "topk"
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

print(f"Compressing all target blocks at {HIDDEN_FRACTION:.4f} hidden fraction (6x)...")
budget = CompressionBudgetSpec(
    hidden_fraction=HIDDEN_FRACTION,
    selection_mode=SELECTION_MODE,
)
full_pack = derive_ffn_compression(atlas, budget)
compressed_by_id = {layer.layer_id: layer for layer in full_pack.layers}

# ── Accumulation curve ────────────────────────────────────────────────────────
print("\nEasiest-first accumulation (6x per block):")
print(f"{'k':>3} {'blocks':>20} {'PPL':>8} {'rel_pct':>8}")
print("-" * 45)

accumulation_results = []

for k in range(1, len(TARGET_BLOCKS) + 1):
    active_blocks = TARGET_BLOCKS[:k]
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

    block_stats = []
    for bid in active_blocks:
        cl = compressed_by_id[bid]
        ratio = cl.source_param_count / max(cl.compressed_param_count, 1)
        block_stats.append({"block": bid, "ratio": float(ratio)})

    print(f"  {k:>2}  {str(active_blocks):>22}  {ppl:8.2f}  {rel:+7.2f}%")

    accumulation_results.append({
        "k": k,
        "blocks": active_blocks,
        "ppl": float(ppl),
        "rel_pct": float(rel),
        "blocks_stats": block_stats,
    })

    del student

# ── Load cycle 033 reference ─────────────────────────────────────────────────
ref_path = RESULTS_DIR / "cycle_033_accumulation.json"
ref_data = None
if ref_path.exists():
    with open(ref_path) as f:
        ref_data = json.load(f)
    print(f"\nLoaded cycle 033 reference (baseline={ref_data['baseline_ppl']:.2f})")

# ── Verdict ───────────────────────────────────────────────────────────────────
ppl_at_4 = next(r["ppl"] for r in accumulation_results if r["k"] == 4)
rel_at_4 = (ppl_at_4 - baseline_ppl) / baseline_ppl * 100
avg_ratio = sum(s["ratio"] for s in accumulation_results[3]["blocks_stats"]) / 4

print(f"\nk=4: PPL={ppl_at_4:.2f}, rel={rel_at_4:+.2f}%, avg_ratio={avg_ratio:.1f}x")

if rel_at_4 <= 8.0 and avg_ratio > 4.0:
    verdict = "confirmed"
    print(f"Verdict: CONFIRMED — k=4 PPL {rel_at_4:+.2f}% <= 8% with {avg_ratio:.1f}x avg compression")
elif rel_at_4 > 12.0:
    verdict = "refuted"
    print(f"Verdict: REFUTED — k=4 PPL {rel_at_4:+.2f}% > 12% (super-additive breakdown)")
else:
    verdict = "inconclusive"
    print(f"Verdict: INCONCLUSIVE — k=4 PPL {rel_at_4:+.2f}% in 8-12% range")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ks = [r["k"] for r in accumulation_results]
rels = [r["rel_pct"] for r in accumulation_results]
ax.plot(ks, rels, marker="o", color="steelblue", ms=7, lw=2, label=f"6× per block (this run)")

if ref_data:
    ref_ks = [r["k"] for r in ref_data["accumulation"]]
    ref_rels = [r["rel_pct"] for r in ref_data["accumulation"]]
    ax.plot(ref_ks, ref_rels, marker="s", color="orange", ms=7, lw=2, ls="--",
            label=f"4× per block (cycle 033)")

ax.axhline(8.0, color="green", ls="--", alpha=0.7, label="+8% threshold (confirmed)")
ax.axhline(12.0, color="red", ls="--", alpha=0.7, label="+12% threshold (refuted)")
ax.axhline(0.0, color="black", ls="-", alpha=0.3)

for r in accumulation_results:
    ax.annotate(f"{r['rel_pct']:+.1f}%", (r["k"], r["rel_pct"]),
                textcoords="offset points", xytext=(6, -10), fontsize=8, color="steelblue")

ax.set_xlabel("Number of blocks compressed (k)")
ax.set_ylabel("Relative PPL change (%)")
ax.set_title(f"Cycle 036: Heterogeneous (6×) vs Uniform (4×) Multi-Block Accumulation\n"
             f"Blocks: {TARGET_BLOCKS}  |  k=4 rel: {rel_at_4:+.1f}%  |  Verdict: {verdict.upper()}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(ks)
plt.tight_layout()

out_png = RESULTS_DIR / "cycle_036_accumulation_6x.png"
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
    "baseline_ppl": float(baseline_ppl),
    "accumulation": accumulation_results,
    "uniform_4x_ref": ref_data["accumulation"] if ref_data else None,
    "verdict": verdict,
    "key_metrics": {
        "rel_pct_k4": float(rel_at_4),
        "avg_ratio_k4": float(avg_ratio),
    },
}

out_json = RESULTS_DIR / "cycle_036_accumulation_6x.json"
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {out_json}")
