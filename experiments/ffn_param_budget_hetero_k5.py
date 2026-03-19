"""Cycle 037: Total FFN parameter reduction budget at <10% PPL — GPT-2 FFN.

Part A: k=4 blocks [7,5,4,1] at 6×, 2 seeds — confirm budget baseline.
Part B: k=5 heterogeneous — base 4 at 6×, 5th block (3 or 6) at 2×/3×/4×, 2 seeds.
Computes total FFN param reduction percentage for each config.
"""
from __future__ import annotations

import copy
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

BASE_4_BLOCKS = [7, 5, 4, 1]  # k=4 easiest
FIFTH_BLOCK_CANDIDATES = [3, 6]
FIFTH_BLOCK_RATIOS = [2, 3, 4]  # lower ratios for 5th block
SEEDS = [42, 123]
BASE_HIDDEN_FRACTION = 1 / 6  # 6× for base 4
PROBE_TRAIN_SEQ = 200
PROBE_HOLDOUT_SEQ = 50
BATCH_SIZE = 4
N_BLOCKS_TOTAL = 12
FFN_HIDDEN = 3072
FFN_IN = 768


def eval_ppl(model, teacher, corpus, batch_size: int) -> float:
    metrics = _evaluate_model(teacher, model, corpus, batch_size=batch_size)
    return float(metrics["perplexity"])


def compute_param_reduction(pack, n_blocks_total=12):
    """Compute total FFN param reduction percentage."""
    # Per-block source params (same for all GPT-2 blocks)
    source_per_block = pack.layers[0].source_param_count
    total_source = source_per_block * n_blocks_total

    # Compressed blocks
    compressed_total = sum(l.compressed_param_count for l in pack.layers)
    # Uncompressed blocks
    n_uncompressed = n_blocks_total - len(pack.layers)
    uncompressed_total = source_per_block * n_uncompressed

    new_total = compressed_total + uncompressed_total
    reduction_pct = (1 - new_total / total_source) * 100
    return float(reduction_pct), int(total_source), int(new_total)


# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading GPT-2...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)

all_blocks_needed = sorted(set(BASE_4_BLOCKS + FIFTH_BLOCK_CANDIDATES))

results = {"part_a": [], "part_b": [], "baselines": {}}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)

    corpus_spec = ProbeCorpusSpec(
        teacher=teacher,
        text_path=SHAKESPEARE_PATH,
        sequence_length=128,
        probe_train_sequences=PROBE_TRAIN_SEQ,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQ,
        eval_val_sequences=PROBE_HOLDOUT_SEQ,
        dataset_name="proxy_shakespeare",
        shuffle_seed=seed,
    )
    corpus = build_probe_corpus(corpus_spec)

    baseline_ppl = eval_ppl(teacher.model, teacher, corpus, BATCH_SIZE)
    results["baselines"][str(seed)] = float(baseline_ppl)
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    print(f"Probing geometry for blocks {all_blocks_needed}...")
    atlas = probe_ffn_geometry(
        teacher, corpus, layer_ids=tuple(all_blocks_needed),
        max_tokens_per_layer=2048, batch_size=BATCH_SIZE, jacobian_samples=8,
    )

    # ── Part A: k=4 at 6× ─────────────────────────────────────────────────
    print("\n--- Part A: k=4 at 6× ---")
    budget_a = CompressionBudgetSpec(hidden_fraction=BASE_HIDDEN_FRACTION, selection_mode="topk")

    # Build sub-atlas for base 4 blocks
    sub_layers_a = tuple(lg for lg in atlas.layer_geometries if lg.layer_id in BASE_4_BLOCKS)
    sub_atlas_a = FfnGeometryAtlas(
        model_family=atlas.model_family, dataset_name=atlas.dataset_name,
        sequence_length=atlas.sequence_length, layer_geometries=sub_layers_a,
        metadata=atlas.metadata,
    )
    pack_a = derive_ffn_compression(sub_atlas_a, budget_a)
    student_a = swap_compressed_ffns(teacher, pack_a, inplace=False)
    ppl_a = eval_ppl(student_a, teacher, corpus, BATCH_SIZE)
    rel_a = (ppl_a - baseline_ppl) / baseline_ppl * 100
    reduction_a, total_src, total_new = compute_param_reduction(pack_a)
    del student_a

    print(f"  k=4@6×: PPL={ppl_a:.2f} ({rel_a:+.2f}%), param_reduction={reduction_a:.1f}%")
    results["part_a"].append({
        "seed": seed, "ppl": float(ppl_a), "rel_pct": float(rel_a),
        "param_reduction_pct": float(reduction_a),
        "total_source_params": total_src, "total_new_params": total_new,
    })

    # ── Part B: k=5 heterogeneous ──────────────────────────────────────────
    print("\n--- Part B: k=5 heterogeneous ---")
    for fifth_block in FIFTH_BLOCK_CANDIDATES:
        for fifth_ratio in FIFTH_BLOCK_RATIOS:
            fifth_hidden = FFN_HIDDEN // fifth_ratio
            config_label = f"k5_block{fifth_block}_{fifth_ratio}x"

            # All 5 blocks in atlas
            k5_blocks = BASE_4_BLOCKS + [fifth_block]
            sub_layers_b = tuple(lg for lg in atlas.layer_geometries if lg.layer_id in k5_blocks)
            sub_atlas_b = FfnGeometryAtlas(
                model_family=atlas.model_family, dataset_name=atlas.dataset_name,
                sequence_length=atlas.sequence_length, layer_geometries=sub_layers_b,
                metadata=atlas.metadata,
            )

            budget_b = CompressionBudgetSpec(
                hidden_fraction=BASE_HIDDEN_FRACTION,
                selection_mode="topk",
                per_layer_hidden={fifth_block: fifth_hidden},
            )
            pack_b = derive_ffn_compression(sub_atlas_b, budget_b)
            student_b = swap_compressed_ffns(teacher, pack_b, inplace=False)
            ppl_b = eval_ppl(student_b, teacher, corpus, BATCH_SIZE)
            rel_b = (ppl_b - baseline_ppl) / baseline_ppl * 100
            reduction_b, _, _ = compute_param_reduction(pack_b)
            del student_b

            print(f"  {config_label}: PPL={ppl_b:.2f} ({rel_b:+.2f}%), reduction={reduction_b:.1f}%")
            results["part_b"].append({
                "seed": seed, "config": config_label,
                "fifth_block": fifth_block, "fifth_ratio": fifth_ratio,
                "ppl": float(ppl_b), "rel_pct": float(rel_b),
                "param_reduction_pct": float(reduction_b),
            })

# ── Aggregate ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("AGGREGATE RESULTS")
print(f"{'='*60}")

# Part A
a_rels = [r["rel_pct"] for r in results["part_a"]]
a_mean = float(np.mean(a_rels))
a_std = float(np.std(a_rels, ddof=1)) if len(a_rels) > 1 else 0.0
a_reduction = results["part_a"][0]["param_reduction_pct"]
print(f"\nPart A — k=4@6×: mean_rel={a_mean:+.2f}% ±{a_std:.2f}%, param_reduction={a_reduction:.1f}%")

# Part B
print(f"\nPart B — k=5 heterogeneous:")
print(f"{'config':>25} {'mean_rel%':>10} {'std':>6} {'reduction%':>12}")
print("-" * 60)

b_summaries = {}
for fifth_block in FIFTH_BLOCK_CANDIDATES:
    for fifth_ratio in FIFTH_BLOCK_RATIOS:
        config = f"k5_block{fifth_block}_{fifth_ratio}x"
        entries = [r for r in results["part_b"] if r["config"] == config]
        rels = [r["rel_pct"] for r in entries]
        mean_rel = float(np.mean(rels))
        std_rel = float(np.std(rels, ddof=1)) if len(rels) > 1 else 0.0
        reduction = entries[0]["param_reduction_pct"]
        seed_disagree = abs(rels[0] - rels[1]) if len(rels) == 2 else 0.0

        b_summaries[config] = {
            "mean_rel": mean_rel, "std_rel": std_rel,
            "reduction": reduction, "seed_disagree": seed_disagree,
        }
        flag = " ⚠DISAGREE" if seed_disagree > 2.0 else ""
        print(f"  {config:>23}  {mean_rel:+9.2f}%  {std_rel:5.2f}  {reduction:11.1f}%{flag}")

# ── Verdict ───────────────────────────────────────────────────────────────────
k4_safe = a_mean <= 10.0
k4_reduction_ok = a_reduction >= 25.0
any_k5_under_12 = any(s["mean_rel"] <= 12.0 and s["seed_disagree"] <= 2.0
                       for s in b_summaries.values())
all_k5_over_15 = all(s["mean_rel"] > 15.0 for s in b_summaries.values())
any_seed_disagree = any(s["seed_disagree"] > 2.0 for s in b_summaries.values())

best_k5 = min(b_summaries.items(), key=lambda x: x[1]["mean_rel"])
print(f"\nBest k=5 config: {best_k5[0]} at {best_k5[1]['mean_rel']:+.2f}%, reduction={best_k5[1]['reduction']:.1f}%")

if k4_safe and k4_reduction_ok and any_k5_under_12:
    verdict = "confirmed"
    print(f"Verdict: CONFIRMED — k=4@6× {a_mean:+.2f}% (<10%), reduction={a_reduction:.1f}% (≥25%), k=5 <12%")
elif not k4_safe or all_k5_over_15:
    verdict = "refuted"
    reason = "k=4 exceeds 10%" if not k4_safe else "all k=5 exceed 15%"
    print(f"Verdict: REFUTED — {reason}")
else:
    verdict = "inconclusive"
    print(f"Verdict: INCONCLUSIVE — k=4 safe but k=5 configs in grey zone or ordering-dependent")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: PPL bars
configs = ["k4@6×"] + list(b_summaries.keys())
means = [a_mean] + [b_summaries[c]["mean_rel"] for c in list(b_summaries.keys())]
stds = [a_std] + [b_summaries[c]["std_rel"] for c in list(b_summaries.keys())]
colors = ["steelblue"] + ["coral" if "block3" in c else "mediumseagreen" for c in b_summaries.keys()]

x = np.arange(len(configs))
ax1.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.8)
ax1.axhline(10.0, color="orange", ls="--", alpha=0.7, label="10% boundary")
ax1.axhline(12.0, color="red", ls="--", alpha=0.7, label="12% confirm threshold")
ax1.set_xticks(x)
ax1.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
ax1.set_ylabel("Mean Relative PPL (%)")
ax1.set_title("PPL by Configuration")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis="y")

# Right: param reduction bars
reductions = [a_reduction] + [b_summaries[c]["reduction"] for c in b_summaries.keys()]
ax2.bar(x, reductions, color=colors, alpha=0.8)
ax2.axhline(25.0, color="green", ls="--", alpha=0.7, label="25% target")
ax2.set_xticks(x)
ax2.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
ax2.set_ylabel("Total FFN Param Reduction (%)")
ax2.set_title("Parameter Savings")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle(f"Cycle 037: FFN Parameter Budget — Verdict: {verdict.upper()}", fontsize=13)
plt.tight_layout()

out_png = RESULTS_DIR / "cycle_037_param_budget.png"
plt.savefig(out_png, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_png}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
results["part_a_summary"] = {"mean_rel": a_mean, "std_rel": a_std, "param_reduction": a_reduction}
results["part_b_summaries"] = b_summaries
results["verdict"] = verdict
results["best_k5"] = {"config": best_k5[0], **best_k5[1]}

out_json = RESULTS_DIR / "cycle_037_param_budget.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out_json}")
