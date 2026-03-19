"""Cycle 053: Per-block corpus sensitivity isolation — GPT-2 FFN.

For each block in [7, 5, 4, 1]:
  - Compress that block alone using Shakespeare probes
  - Compress that block alone using WikiText-2 probes
  - Evaluate each compressed model on BOTH corpora
  - Compute per-block sensitivity delta = (wt2_rel_ppl) - (shk_rel_ppl)

Key question: Is sensitivity concentrated in one/two blocks (>=50% of total delta)?
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
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
WIKITEXT_PATH = str(Path(__file__).parent.parent / "data" / "wikitext2_valid.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BLOCKS = [7, 5, 4, 1]
HIDDEN_FRACTION = 1 / 6  # 6×
PROBE_TRAIN_SEQ = 200
PROBE_EVAL_SEQ = 200
BATCH_SIZE = 4
SEED = 42


def eval_ppl(model, teacher, corpus, batch_size: int) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=batch_size)
    return float(metrics["perplexity"])


# ── Load teacher ──────────────────────────────────────────────────────────────
print("Loading GPT-2...")
torch.manual_seed(SEED)
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

# ── Build both corpora ────────────────────────────────────────────────────────
print("Building Shakespeare corpus...")
shk_spec = ProbeCorpusSpec(
    teacher=teacher,
    text_path=SHAKESPEARE_PATH,
    sequence_length=128,
    probe_train_sequences=PROBE_TRAIN_SEQ,
    probe_holdout_sequences=50,
    eval_val_sequences=PROBE_EVAL_SEQ,
    dataset_name="shakespeare",
    shuffle_seed=SEED,
)
shk_corpus = build_probe_corpus(shk_spec)

print("Building WikiText-2 corpus...")
wiki_spec = ProbeCorpusSpec(
    teacher=teacher,
    text_path=WIKITEXT_PATH,
    sequence_length=128,
    probe_train_sequences=PROBE_TRAIN_SEQ,
    probe_holdout_sequences=50,
    eval_val_sequences=PROBE_EVAL_SEQ,
    dataset_name="wikitext2",
    shuffle_seed=SEED,
)
wiki_corpus = build_probe_corpus(wiki_spec)

# ── Teacher baselines ─────────────────────────────────────────────────────────
print("Computing teacher baselines...")
shk_baseline = eval_ppl(teacher.model, teacher, shk_corpus, BATCH_SIZE)
wt2_baseline = eval_ppl(teacher.model, teacher, wiki_corpus, BATCH_SIZE)
print(f"  Shakespeare baseline: {shk_baseline:.2f}")
print(f"  WikiText-2 baseline:  {wt2_baseline:.2f}")

# ── Budget spec ───────────────────────────────────────────────────────────────
budget = CompressionBudgetSpec(
    hidden_fraction=HIDDEN_FRACTION,
    selection_mode="topk",
    kmeans_seed=SEED,
)

# ── Per-block experiment ──────────────────────────────────────────────────────
results = {
    "metadata": {
        "target_blocks": TARGET_BLOCKS,
        "hidden_fraction": HIDDEN_FRACTION,
        "probe_train_seq": PROBE_TRAIN_SEQ,
        "probe_eval_seq": PROBE_EVAL_SEQ,
        "seed": SEED,
    },
    "baselines": {
        "shk_ppl": float(shk_baseline),
        "wt2_ppl": float(wt2_baseline),
    },
    "blocks": {},
}

probe_corpora = {
    "shakespeare": shk_corpus,
    "wikitext2": wiki_corpus,
}

for block_id in TARGET_BLOCKS:
    print(f"\n{'='*60}")
    print(f"Block {block_id}")
    print(f"{'='*60}")

    block_results = {}

    for probe_name, probe_corpus in probe_corpora.items():
        print(f"\n  Probe corpus: {probe_name}")

        # Probe geometry for this single block only
        atlas = probe_ffn_geometry(
            teacher,
            probe_corpus,
            layer_ids=(block_id,),
            max_tokens_per_layer=2048,
            batch_size=BATCH_SIZE,
            jacobian_samples=8,
        )

        # Derive compression
        pack = derive_ffn_compression(atlas, budget)

        # Swap only this block into teacher model
        student_model = swap_compressed_ffns(teacher, pack, inplace=False)

        # Evaluate on both corpora
        shk_ppl = eval_ppl(student_model, teacher, shk_corpus, BATCH_SIZE)
        wt2_ppl = eval_ppl(student_model, teacher, wiki_corpus, BATCH_SIZE)

        shk_rel = (shk_ppl / shk_baseline - 1.0) * 100.0
        wt2_rel = (wt2_ppl / wt2_baseline - 1.0) * 100.0
        sensitivity_delta = wt2_rel - shk_rel

        print(f"    Shakespeare eval: {shk_ppl:.2f} ({shk_rel:+.2f}%)")
        print(f"    WikiText-2 eval:  {wt2_ppl:.2f} ({wt2_rel:+.2f}%)")
        print(f"    Sensitivity delta: {sensitivity_delta:+.2f}pp")

        block_results[probe_name] = {
            "shk_ppl": float(shk_ppl),
            "wt2_ppl": float(wt2_ppl),
            "shk_rel_pct": float(shk_rel),
            "wt2_rel_pct": float(wt2_rel),
            "sensitivity_delta": float(sensitivity_delta),
        }

        del student_model, pack, atlas

    results["blocks"][str(block_id)] = block_results

# ── Summary analysis ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

for probe_name in ["shakespeare", "wikitext2"]:
    print(f"\nProbe corpus: {probe_name}")
    deltas = []
    for block_id in TARGET_BLOCKS:
        delta = results["blocks"][str(block_id)][probe_name]["sensitivity_delta"]
        deltas.append(delta)
        print(f"  Block {block_id}: sensitivity delta = {delta:+.2f}pp")

    total_delta = sum(abs(d) for d in deltas)
    if total_delta > 0:
        for i, block_id in enumerate(TARGET_BLOCKS):
            frac = abs(deltas[i]) / total_delta * 100
            print(f"  Block {block_id}: {frac:.1f}% of total absolute sensitivity")

    max_delta = max(abs(d) for d in deltas)
    max_block = TARGET_BLOCKS[np.argmax([abs(d) for d in deltas])]
    concentration = max_delta / total_delta * 100 if total_delta > 0 else 0
    print(f"  Most sensitive block: {max_block} ({concentration:.1f}% of total)")

# Save results
out_path = RESULTS_DIR / "cycle_053_per_block_sensitivity.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {out_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(len(TARGET_BLOCKS))
width = 0.35

for ax_idx, probe_name in enumerate(["shakespeare", "wikitext2"]):
    ax = axes[ax_idx]

    shk_rels = [results["blocks"][str(b)][probe_name]["shk_rel_pct"] for b in TARGET_BLOCKS]
    wt2_rels = [results["blocks"][str(b)][probe_name]["wt2_rel_pct"] for b in TARGET_BLOCKS]
    deltas = [results["blocks"][str(b)][probe_name]["sensitivity_delta"] for b in TARGET_BLOCKS]

    bars1 = ax.bar(x - width / 2, shk_rels, width, label="Shakespeare eval", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + width / 2, wt2_rels, width, label="WikiText-2 eval", color="#FF9800", alpha=0.8)

    ax.set_xlabel("Block ID")
    ax.set_ylabel("Relative PPL increase (%)")
    ax.set_title(f"Probe corpus: {probe_name}\n(single-block 6x compression)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Block {b}" for b in TARGET_BLOCKS])
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    # Annotate sensitivity deltas
    for i, (xi, delta) in enumerate(zip(x, deltas)):
        ymax = max(shk_rels[i], wt2_rels[i])
        ax.text(xi, ymax + 0.5, f"Δ={delta:+.1f}", ha="center", va="bottom",
                fontsize=9, color="darkred", fontweight="bold")

plt.suptitle("Per-Block Corpus Sensitivity: Shakespeare vs WikiText-2 Probes\n"
             "GPT-2, blocks [7,5,4,1], 6× compression (topk)", fontsize=12)
plt.tight_layout()
plot_path = RESULTS_DIR / "cycle_053_per_block_sensitivity.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {plot_path}")

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("VERDICT ANALYSIS")
print(f"{'='*60}")

for probe_name in ["shakespeare", "wikitext2"]:
    deltas = [results["blocks"][str(b)][probe_name]["sensitivity_delta"] for b in TARGET_BLOCKS]
    total = sum(abs(d) for d in deltas)
    max_frac = max(abs(d) for d in deltas) / total * 100 if total > 0 else 0
    min_delta = min(abs(d) for d in deltas)
    max_delta = max(abs(d) for d in deltas)
    mean_delta = sum(abs(d) for d in deltas) / len(deltas)
    within_25pct = all(abs(abs(d) - mean_delta) / mean_delta < 0.25 for d in deltas) if mean_delta > 0 else False

    if max_frac >= 50:
        verdict = "CONFIRMED"
        reason = f"max block = {max_frac:.1f}% >= 50% of total"
    elif within_25pct:
        verdict = "REFUTED"
        reason = "all blocks within +/-25% of mean delta"
    else:
        verdict = "INCONCLUSIVE"
        reason = f"max block = {max_frac:.1f}%, not uniform"

    print(f"\n  Probe={probe_name}: {verdict} ({reason})")
    print(f"  Deltas: {[f'{d:+.2f}' for d in deltas]}")
    print(f"  Max fraction of total: {max_frac:.1f}%")

print("\nDone.")
