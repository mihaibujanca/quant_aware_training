"""Cycle 040: Compression ceiling scan — 8x and 10x single-block — GPT-2 FFN.

Sweeps {6x, 8x, 10x} on blocks [7, 5, 4] with 3 seeds and 200 eval sequences.
6x serves as cross-check against cycle 035.
"""
from __future__ import annotations

import json
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

TARGET_BLOCKS = [7, 5, 4]
RATIOS = [6, 8, 10]
SEEDS = [42, 123, 7]
PROBE_TRAIN_SEQ = 200
PROBE_HOLDOUT_SEQ = 200
BATCH_SIZE = 4


def eval_ppl(model, teacher, corpus, batch_size: int) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=batch_size)
    return float(metrics["perplexity"])


# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

all_results = []
baselines = {}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)

    corpus_spec = ProbeCorpusSpec(
        teacher=teacher, text_path=SHAKESPEARE_PATH, sequence_length=128,
        probe_train_sequences=PROBE_TRAIN_SEQ, probe_holdout_sequences=PROBE_HOLDOUT_SEQ,
        eval_val_sequences=PROBE_HOLDOUT_SEQ, dataset_name="proxy_shakespeare",
        shuffle_seed=seed,
    )
    corpus = build_probe_corpus(corpus_spec)

    baseline_ppl = eval_ppl(teacher.model, teacher, corpus, BATCH_SIZE)
    baselines[str(seed)] = float(baseline_ppl)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    print(f"Probing geometry for blocks {TARGET_BLOCKS}...")
    atlas = probe_ffn_geometry(
        teacher, corpus, layer_ids=tuple(TARGET_BLOCKS),
        max_tokens_per_layer=2048, batch_size=BATCH_SIZE, jacobian_samples=8,
    )

    print(f"\n{'block':>6} {'ratio':>6} {'PPL':>8} {'rel%':>8}")
    print("-" * 35)

    for ratio in RATIOS:
        budget = CompressionBudgetSpec(
            hidden_fraction=1.0/ratio, selection_mode="topk",
        )
        pack = derive_ffn_compression(atlas, budget)

        for block_id in TARGET_BLOCKS:
            cl = next(l for l in pack.layers if l.layer_id == block_id)
            sub_pack = CompressedFfnPack(
                method=pack.method, model_family=pack.model_family,
                dataset_name=pack.dataset_name, layers=(cl,), metadata=pack.metadata,
            )
            student = swap_compressed_ffns(teacher, sub_pack, inplace=False)
            ppl = eval_ppl(student, teacher, corpus, BATCH_SIZE)
            rel = (ppl - baseline_ppl) / baseline_ppl * 100
            del student

            print(f"  {block_id:>4}  {ratio:>4}x  {ppl:8.2f}  {rel:+7.2f}%")

            all_results.append({
                "seed": seed, "block": block_id, "ratio": ratio,
                "baseline_ppl": float(baseline_ppl),
                "ppl": float(ppl), "rel_pct": float(rel),
            })

# ── Aggregate ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("AGGREGATE RESULTS")
print(f"{'='*60}")
print(f"\n{'block':>6} {'ratio':>6} {'mean%':>8} {'std%':>6}")
print("-" * 30)

summaries = {}
for block_id in TARGET_BLOCKS:
    for ratio in RATIOS:
        entries = [r for r in all_results if r["block"] == block_id and r["ratio"] == ratio]
        rels = [r["rel_pct"] for r in entries]
        mean_rel = float(np.mean(rels))
        std_rel = float(np.std(rels, ddof=1)) if len(rels) > 1 else 0.0
        key = f"b{block_id}_{ratio}x"
        summaries[key] = {"mean": mean_rel, "std": std_rel, "rels": rels}
        print(f"  {block_id:>4}  {ratio:>4}x  {mean_rel:+7.2f}%  {std_rel:5.2f}")

# ── Verdict ───────────────────────────────────────────────────────────────────
blocks_under_10_at_8x = []
for block_id in TARGET_BLOCKS:
    key = f"b{block_id}_8x"
    if summaries[key]["mean"] <= 10.0:
        blocks_under_10_at_8x.append(block_id)

print(f"\nBlocks with 8x mean ≤10%: {blocks_under_10_at_8x}")

if len(blocks_under_10_at_8x) >= 1:
    verdict = "confirmed"
    print(f"Verdict: CONFIRMED — {len(blocks_under_10_at_8x)} block(s) tolerate 8x at ≤10%")
elif len(blocks_under_10_at_8x) == 0:
    # Check if all exceed 10%
    verdict = "refuted"
    print(f"Verdict: REFUTED — all 3 blocks exceed 10% at 8x")

# Cross-check: 6x vs cycle 035
print("\n6x cross-check vs cycle 035:")
for block_id in TARGET_BLOCKS:
    key = f"b{block_id}_6x"
    print(f"  Block {block_id}: this run mean={summaries[key]['mean']:+.2f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(TARGET_BLOCKS))
width = 0.25
colors = {"6": "steelblue", "8": "coral", "10": "darkred"}

for i, ratio in enumerate(RATIOS):
    means = [summaries[f"b{b}_{ratio}x"]["mean"] for b in TARGET_BLOCKS]
    stds = [summaries[f"b{b}_{ratio}x"]["std"] for b in TARGET_BLOCKS]
    bars = ax.bar(x + i*width, means, width, yerr=stds, capsize=3,
                  label=f"{ratio}×", color=colors[str(ratio)], alpha=0.8)
    for j, (m, s) in enumerate(zip(means, stds)):
        ax.annotate(f"{m:+.1f}%", (x[j] + i*width, m),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

ax.axhline(10.0, color="red", ls="--", alpha=0.7, label="10% threshold")
ax.axhline(0, color="black", ls="-", alpha=0.3)
ax.set_xticks(x + width)
ax.set_xticklabels([f"Block {b}" for b in TARGET_BLOCKS])
ax.set_ylabel("Mean Relative PPL change (%)")
ax.set_title(f"Cycle 040: Compression Ceiling Scan (3 seeds, 200 eval seqs)\n"
             f"Blocks at ≤10% at 8×: {blocks_under_10_at_8x}  |  Verdict: {verdict.upper()}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()

out_png = RESULTS_DIR / "cycle_040_ceiling_scan.png"
plt.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "config": {"target_blocks": TARGET_BLOCKS, "ratios": RATIOS, "seeds": SEEDS},
    "baselines": baselines,
    "per_seed_results": all_results,
    "summaries": summaries,
    "blocks_under_10_at_8x": blocks_under_10_at_8x,
    "verdict": verdict,
}

out_json = RESULTS_DIR / "cycle_040_ceiling_scan.json"
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {out_json}")
