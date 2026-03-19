"""Cycle 041: GPT-2-medium single-block 4x generalization probe.

Tests whether topk + lstsq refit transfers from GPT-2-small to GPT-2-medium.
6 evenly spaced blocks, 3 seeds, 200 eval sequences.
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

TARGET_BLOCKS = [0, 4, 8, 12, 16, 20]
RATIO = 4
HIDDEN_FRACTION = 1.0 / RATIO
SEEDS = [42, 123, 7]
PROBE_TRAIN_SEQ = 200
PROBE_HOLDOUT_SEQ = 200
BATCH_SIZE = 2  # smaller batch for larger model
SEED = 42


def eval_ppl(model, teacher, corpus, batch_size: int) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=batch_size)
    return float(metrics["perplexity"])


# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading GPT-2-medium...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2-medium")
teacher = load_nanochat_teacher(load_spec)
print(f"  device: {teacher.device}")

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

    # Probe one block at a time to manage memory
    for block_id in TARGET_BLOCKS:
        print(f"\n  Block {block_id}...")
        atlas = probe_ffn_geometry(
            teacher, corpus, layer_ids=(block_id,),
            max_tokens_per_layer=2048, batch_size=BATCH_SIZE, jacobian_samples=8,
        )

        budget = CompressionBudgetSpec(
            hidden_fraction=HIDDEN_FRACTION, selection_mode="topk",
        )
        pack = derive_ffn_compression(atlas, budget)
        cl = pack.layers[0]
        actual_ratio = cl.source_param_count / max(cl.compressed_param_count, 1)

        student = swap_compressed_ffns(teacher, pack, inplace=False)
        ppl = eval_ppl(student, teacher, corpus, BATCH_SIZE)
        rel = (ppl - baseline_ppl) / baseline_ppl * 100
        del student

        print(f"    PPL={ppl:.2f} ({rel:+.2f}%), ratio={actual_ratio:.1f}x")

        all_results.append({
            "seed": seed, "block": block_id, "ratio": float(actual_ratio),
            "baseline_ppl": float(baseline_ppl),
            "ppl": float(ppl), "rel_pct": float(rel),
        })

# ── Aggregate ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("AGGREGATE RESULTS (GPT-2-medium, 4x)")
print(f"{'='*60}")
print(f"\n{'block':>6} {'mean%':>8} {'std%':>6}")
print("-" * 25)

summaries = {}
for block_id in TARGET_BLOCKS:
    entries = [r for r in all_results if r["block"] == block_id]
    rels = [r["rel_pct"] for r in entries]
    mean_rel = float(np.mean(rels))
    std_rel = float(np.std(rels, ddof=1)) if len(rels) > 1 else 0.0
    summaries[str(block_id)] = {"mean": mean_rel, "std": std_rel, "rels": rels}
    print(f"  {block_id:>4}  {mean_rel:+7.2f}%  {std_rel:5.2f}")

# ── Verdict ───────────────────────────────────────────────────────────────────
blocks_under_10 = [b for b in TARGET_BLOCKS if summaries[str(b)]["mean"] <= 10.0]
n_pass = len(blocks_under_10)

print(f"\nBlocks with mean ≤10%: {n_pass}/6 — {blocks_under_10}")

if n_pass >= 4:
    verdict = "confirmed"
    print(f"Verdict: CONFIRMED — {n_pass}/6 blocks pass (≥4 required)")
elif n_pass <= 1:
    verdict = "refuted"
    print(f"Verdict: REFUTED — only {n_pass}/6 blocks pass (≤1)")
else:
    verdict = "inconclusive"
    print(f"Verdict: INCONCLUSIVE — {n_pass}/6 blocks pass (2-3)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(TARGET_BLOCKS))
means = [summaries[str(b)]["mean"] for b in TARGET_BLOCKS]
stds = [summaries[str(b)]["std"] for b in TARGET_BLOCKS]
colors = ["green" if m <= 10 else "red" for m in means]

ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.8)
ax.axhline(10.0, color="red", ls="--", alpha=0.7, label="10% threshold")
ax.axhline(0, color="black", ls="-", alpha=0.3)

for i, (m, s) in enumerate(zip(means, stds)):
    ax.annotate(f"{m:+.1f}%\n±{s:.1f}", (i, m),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([f"Block {b}" for b in TARGET_BLOCKS])
ax.set_ylabel("Mean Relative PPL change (%)")
ax.set_title(f"Cycle 041: GPT-2-Medium Single-Block 4× Compression (3 seeds, 200 eval)\n"
             f"Pass: {n_pass}/6  |  Verdict: {verdict.upper()}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()

out_png = RESULTS_DIR / "cycle_041_medium_generalization.png"
plt.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
result = {
    "config": {
        "model": "gpt2-medium", "n_layers": 24, "d_model": 1024, "ffn_dim": 4096,
        "target_blocks": TARGET_BLOCKS, "ratio": RATIO, "seeds": SEEDS,
    },
    "baselines": baselines,
    "per_seed_results": all_results,
    "summaries": summaries,
    "blocks_under_10pct": blocks_under_10,
    "n_pass": n_pass,
    "verdict": verdict,
}

out_json = RESULTS_DIR / "cycle_041_medium_generalization.json"
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {out_json}")
