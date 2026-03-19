"""Cycle 067: Centroid-merge channel selection quality vs probe corpus size — GPT-2 FFN.

Measures how Jaccard stability of topk channel selection (and downstream PPL)
varies as we scale the probe corpus from 64 to 2048 tokens on GPT-2 block 7.

3 seeds × 6 probe sizes = 18 runs. Fixed 200-sequence eval corpus.
Reference: 2048-token probe. Threshold: Jaccard ≥ 0.85 and PPL within 2% at ≤ 256 tokens.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnPack,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    _evaluate_model,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BLOCK_ID = 7
RATIO = 6
HIDDEN_FRACTION = 1.0 / RATIO   # k ≈ 512 out of 3072
PROBE_SIZES = [64, 128, 256, 512, 1024, 2048]   # tokens for atlas building
SEEDS = [42, 123, 7]
SEQ_LEN = 128                    # sequence length for corpus windows
EVAL_SEQ = 200                   # fixed eval sequences
BATCH_SIZE = 4
JACOBIAN_SAMPLES = 8

SHAKESPEARE_PATH = str(Path(__file__).parent.parent / "data" / "shakespeare.txt")
RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = RESULTS_DIR / "cycle_067_centroid_merge_corpus_size.json"
PLOT_PATH = RESULTS_DIR / "cycle_067_centroid_merge_corpus_size.png"

# ---------------------------------------------------------------------------
# Load model once
# ---------------------------------------------------------------------------
print("Loading GPT-2...")
teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
print(f"  device={teacher.device}, activation={teacher.activation_kind}")


def eval_ppl(model, corpus) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=BATCH_SIZE)
    return float(metrics["perplexity"])


# ---------------------------------------------------------------------------
# Build a single fixed eval corpus (seed=42, 200 sequences, used for all PPL evals)
# ---------------------------------------------------------------------------
print("Building fixed eval corpus...")
eval_corpus_spec = ProbeCorpusSpec(
    teacher=teacher,
    text_path=SHAKESPEARE_PATH,
    sequence_length=SEQ_LEN,
    probe_train_sequences=1,      # not used for PPL eval
    probe_holdout_sequences=1,    # not used for PPL eval
    eval_val_sequences=EVAL_SEQ,
    dataset_name="shakespeare",
    shuffle_seed=42,
)
eval_corpus = build_probe_corpus(eval_corpus_spec)
print(f"  eval_val shape: {eval_corpus.eval_val.shape}")

# Baseline GPT-2 PPL on fixed eval corpus
print("Computing baseline PPL...")
baseline_ppl = eval_ppl(teacher.model, eval_corpus)
print(f"  Baseline PPL: {baseline_ppl:.4f}")
assert 20 < baseline_ppl < 200, f"Unexpected baseline PPL {baseline_ppl:.2f} — sanity check failed"

# ---------------------------------------------------------------------------
# Main loop: probe_size × seed
# ---------------------------------------------------------------------------
# per_run[probe_size][seed] = {"channels": sorted list, "ppl": float}
per_run: dict[int, dict[int, dict]] = {ps: {} for ps in PROBE_SIZES}

for probe_size in PROBE_SIZES:
    probe_holdout_seq = max(1, math.ceil(probe_size / SEQ_LEN))
    print(f"\n== probe_size={probe_size} tokens (holdout_seq={probe_holdout_seq}) ==")
    for seed in SEEDS:
        torch.manual_seed(seed)
        # Build probe corpus with enough holdout sequences for atlas
        corpus_spec = ProbeCorpusSpec(
            teacher=teacher,
            text_path=SHAKESPEARE_PATH,
            sequence_length=SEQ_LEN,
            probe_train_sequences=probe_holdout_seq,
            probe_holdout_sequences=probe_holdout_seq,
            eval_val_sequences=EVAL_SEQ,
            dataset_name="shakespeare",
            shuffle_seed=seed,
        )
        corpus = build_probe_corpus(corpus_spec)

        # Build atlas capped at probe_size tokens
        atlas = probe_ffn_geometry(
            teacher,
            corpus,
            layer_ids=(BLOCK_ID,),
            max_tokens_per_layer=probe_size,
            batch_size=BATCH_SIZE,
            jacobian_samples=JACOBIAN_SAMPLES,
        )

        # Derive compression (topk, 6x ratio)
        budget = CompressionBudgetSpec(
            hidden_fraction=HIDDEN_FRACTION,
            selection_mode="topk",
        )
        pack = derive_ffn_compression(atlas, budget)

        # Assert channel count
        layer_pack = pack.layers[0]
        k = layer_pack.compressed_hidden_dim
        assert k == layer_pack.selected_indices.shape[0], f"Channel count mismatch: {k} vs {layer_pack.selected_indices.shape[0]}"
        selected_channels = layer_pack.selected_indices.sort().values.tolist()
        print(f"  seed={seed}: k={k} channels selected")

        # Evaluate PPL with this single-block swap
        student = swap_compressed_ffns(teacher, pack, inplace=False)
        ppl = eval_ppl(student, eval_corpus)
        ppl_pct = (ppl - baseline_ppl) / baseline_ppl * 100
        print(f"  seed={seed}: PPL={ppl:.4f}, rel={ppl_pct:+.2f}%")

        per_run[probe_size][seed] = {
            "channels": selected_channels,
            "k": k,
            "ppl": ppl,
            "ppl_pct": ppl_pct,
        }

# ---------------------------------------------------------------------------
# Compute Jaccard (3-way: intersection/union across all 3 seeds)
# and pairwise mean Jaccard for the reviewer's note
# ---------------------------------------------------------------------------
print("\n== Summary ==")
print(f"{'probe_size':>12} {'jaccard_3way':>14} {'jaccard_pair':>14} {'ppl_mean':>10} {'ppl_std':>9}")
summary = []

for probe_size in PROBE_SIZES:
    seed_channels = {seed: set(per_run[probe_size][seed]["channels"]) for seed in SEEDS}
    # 3-way Jaccard
    intersection_3 = seed_channels[SEEDS[0]] & seed_channels[SEEDS[1]] & seed_channels[SEEDS[2]]
    union_3 = seed_channels[SEEDS[0]] | seed_channels[SEEDS[1]] | seed_channels[SEEDS[2]]
    jaccard_3 = len(intersection_3) / len(union_3) if union_3 else 1.0

    # Pairwise Jaccard (mean of 3 pairs)
    pairs = [(SEEDS[0], SEEDS[1]), (SEEDS[0], SEEDS[2]), (SEEDS[1], SEEDS[2])]
    pairwise_jaccards = []
    for s1, s2 in pairs:
        c1, c2 = seed_channels[s1], seed_channels[s2]
        pj = len(c1 & c2) / len(c1 | c2) if (c1 | c2) else 1.0
        pairwise_jaccards.append(pj)
    jaccard_pair_mean = sum(pairwise_jaccards) / len(pairwise_jaccards)

    ppls = [per_run[probe_size][seed]["ppl"] for seed in SEEDS]
    ppl_mean = sum(ppls) / len(ppls)
    ppl_std = (sum((p - ppl_mean) ** 2 for p in ppls) / len(ppls)) ** 0.5
    ppl_pcts = [per_run[probe_size][seed]["ppl_pct"] for seed in SEEDS]
    ppl_pct_mean = sum(ppl_pcts) / len(ppl_pcts)
    ppl_pct_std = (sum((p - ppl_pct_mean) ** 2 for p in ppl_pcts) / len(ppl_pcts)) ** 0.5

    print(f"{probe_size:>12d} {jaccard_3:>14.4f} {jaccard_pair_mean:>14.4f} {ppl_mean:>10.4f} {ppl_std:>9.4f}")
    summary.append({
        "probe_size": probe_size,
        "jaccard_3way": jaccard_3,
        "jaccard_pairwise_mean": jaccard_pair_mean,
        "pairwise_jaccards": pairwise_jaccards,
        "ppl_mean": ppl_mean,
        "ppl_std": ppl_std,
        "ppl_pct_mean": ppl_pct_mean,
        "ppl_pct_std": ppl_pct_std,
        "per_seed": {
            str(seed): {
                "k": per_run[probe_size][seed]["k"],
                "ppl": per_run[probe_size][seed]["ppl"],
                "ppl_pct": per_run[probe_size][seed]["ppl_pct"],
            }
            for seed in SEEDS
        },
    })

# Validation: Jaccard at 2048 should be >0.9; log warning if not
ref_row = next(r for r in summary if r["probe_size"] == 2048)
if ref_row["jaccard_3way"] <= 0.9:
    print(f"\nSelf-consistency WARNING: Jaccard@2048={ref_row['jaccard_3way']:.4f} <= 0.9 (method inherently variable)")
else:
    print(f"\nSelf-consistency check PASSED: Jaccard@2048={ref_row['jaccard_3way']:.4f}")

# Reference PPL: 2048-token mean PPL
ref_ppl = ref_row["ppl_mean"]
print(f"Reference PPL (2048 tokens): {ref_ppl:.4f}")

# Compute PPL relative to 2048-token reference (for threshold adjudication)
for row in summary:
    row["ppl_vs_ref_pct"] = (row["ppl_mean"] - ref_ppl) / ref_ppl * 100

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
output = {
    "cycle": 67,
    "item": 50,
    "block_id": BLOCK_ID,
    "ratio": RATIO,
    "seeds": SEEDS,
    "probe_sizes": PROBE_SIZES,
    "baseline_ppl": baseline_ppl,
    "ref_ppl_2048": ref_ppl,
    "summary": summary,
}
JSON_PATH.write_text(json.dumps(output, indent=2))
print(f"\nSaved JSON: {JSON_PATH}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Centroid-merge channel selection stability vs probe corpus size\nGPT-2 block 7, 6x compression (k≈512 of 3072)", fontsize=11)

probe_x = [r["probe_size"] for r in summary]
jac3 = [r["jaccard_3way"] for r in summary]
jac_pair = [r["jaccard_pairwise_mean"] for r in summary]

ax1.plot(probe_x, jac3, "o-", color="steelblue", label="3-way Jaccard", linewidth=2, markersize=7)
ax1.plot(probe_x, jac_pair, "s--", color="cornflowerblue", label="Pairwise mean Jaccard", linewidth=1.5, markersize=5)
ax1.axhline(0.85, color="red", linestyle="--", linewidth=1.2, label="Threshold (0.85)")
ax1.set_xscale("log")
ax1.set_xlabel("Probe corpus size (tokens)")
ax1.set_ylabel("Jaccard similarity (across 3 seeds)")
ax1.set_title("Channel selection stability")
ax1.legend(fontsize=9)
ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3)

ppl_means = [r["ppl_pct_mean"] for r in summary]
ppl_stds = [r["ppl_pct_std"] for r in summary]
# Also plot PPL vs 2048-token reference
ppl_vs_ref = [r["ppl_vs_ref_pct"] for r in summary]

ax2.errorbar(probe_x, ppl_means, yerr=ppl_stds, fmt="o-", color="darkorange",
             label="PPL% vs baseline GPT-2", linewidth=2, markersize=7, capsize=4)
ax2.plot(probe_x, ppl_vs_ref, "s--", color="tomato", label="PPL% vs 2048-token ref", linewidth=1.5, markersize=5)
ax2.axhline(2.0, color="green", linestyle="--", linewidth=1.2, label="+2% threshold")
ax2.axhline(5.0, color="red", linestyle="--", linewidth=1.2, label="+5% threshold")
ax2.set_xscale("log")
ax2.set_xlabel("Probe corpus size (tokens)")
ax2.set_ylabel("PPL degradation (%)")
ax2.set_title("Perplexity degradation")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved plot: {PLOT_PATH}")

# ---------------------------------------------------------------------------
# Adjudication
# ---------------------------------------------------------------------------
row_256 = next(r for r in summary if r["probe_size"] == 256)
j_at_256 = row_256["jaccard_3way"]
ppl_vs_ref_at_256 = row_256["ppl_vs_ref_pct"]
ppl_std_at_256 = row_256["ppl_pct_std"]

print("\n== Adjudication ==")
print(f"  Jaccard@256:        {j_at_256:.4f}  (threshold: ≥0.85)")
print(f"  PPL vs ref@256:     {ppl_vs_ref_at_256:+.2f}%  (threshold: ≤2% confirmed, >5% refuted)")
print(f"  PPL std@256:        {ppl_std_at_256:.2f}%  (inconclusive if >3%)")

if j_at_256 >= 0.85 and abs(ppl_vs_ref_at_256) <= 2.0:
    verdict = "confirmed"
elif j_at_256 < 0.6 or ppl_vs_ref_at_256 > 5.0:
    verdict = "refuted"
elif ppl_std_at_256 > 3.0 and j_at_256 >= 0.6:
    verdict = "inconclusive"
else:
    verdict = "inconclusive"

print(f"\n  VERDICT: {verdict.upper()}")
print(f"\nDone.")
