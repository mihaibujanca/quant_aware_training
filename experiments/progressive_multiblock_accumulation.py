"""Progressive multi-block FFN accumulation curve (cycle 007, item 12).

Compresses k=1..12 GPT-2 FFN blocks at 16x budget in three orderings:
  - easiest-first (ascending single-block damage from cycle 005)
  - hardest-first (descending)
  - random (seed=42)

Measures full-model perplexity at each step. Proxy-only (gpt2+Shakespeare).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
    _evaluate_model,
)

# ── Config ──────────────────────────────────────────────────────────
TEACHER_SPEC = TeacherLoadSpec(
    source="hf_pretrained",
    hf_model_name="gpt2",
    device="cpu",
)
CORPUS_SPEC_PARTIAL = dict(
    text_path="data/shakespeare.txt",
    sequence_length=128,
    probe_train_sequences=64,
    probe_holdout_sequences=32,
    eval_val_sequences=32,
    dataset_name="proxy_shakespeare",
)
HIDDEN_FRACTION = 0.0625  # 16x compression
RANDOM_SEED = 42
OUTPUT_DIR = Path("autoresearch/results")

# ── Layer ranking from cycle 005 (16x single-block ppl) ────────────
# Measured perplexities at hf=0.0625, teacher ppl=122.350:
# layer: ppl  → relative increase
CYCLE005_PPL = {
    0: 162.867, 1: 123.256, 2: 123.749, 3: 123.570,
    4: 123.132, 5: 122.500, 6: 123.573, 7: 121.581,
    8: 124.739, 9: 124.350, 10: 124.564, 11: 130.322,
}
TEACHER_PPL = 122.350


def rank_layers_by_damage() -> list[int]:
    """Return layer ids sorted by ascending single-block damage (easiest first)."""
    damage = {lid: (ppl - TEACHER_PPL) / TEACHER_PPL for lid, ppl in CYCLE005_PPL.items()}
    return sorted(damage, key=lambda lid: damage[lid])


def build_orderings() -> dict[str, list[int]]:
    easiest = rank_layers_by_damage()
    hardest = list(reversed(easiest))
    rng = random.Random(RANDOM_SEED)
    shuffled = list(range(12))
    rng.shuffle(shuffled)
    return {"easiest_first": easiest, "hardest_first": hardest, "random": shuffled}


def subset_atlas(atlas: FfnGeometryAtlas, layer_ids: set[int]) -> FfnGeometryAtlas:
    """Return a copy of the atlas with only the specified layers."""
    filtered = tuple(lg for lg in atlas.layer_geometries if lg.layer_id in layer_ids)
    return replace(atlas, layer_geometries=filtered)


def main() -> None:
    print("Loading teacher...")
    teacher = load_nanochat_teacher(TEACHER_SPEC)
    corpus = build_probe_corpus(
        ProbeCorpusSpec(teacher=teacher, **CORPUS_SPEC_PARTIAL)
    )

    # Verify teacher perplexity
    teacher_metrics = _evaluate_model(teacher, teacher.model, corpus, batch_size=4)
    teacher_ppl = teacher_metrics["perplexity"]
    print(f"Teacher perplexity: {teacher_ppl:.3f}")

    # Probe all 12 layers once
    print("Probing all 12 FFN layers...")
    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        min_hidden=32,
        selection_mode="topk",
        jacobian_samples=8,
    )
    full_atlas = probe_ffn_geometry(
        teacher, corpus, layer_ids=None,
        max_tokens_per_layer=2048, batch_size=4,
        jacobian_samples=budget.jacobian_samples,
    )
    print(f"Probed {len(full_atlas.layer_geometries)} layers")

    orderings = build_orderings()
    print(f"Orderings: {json.dumps({k: v for k, v in orderings.items()}, indent=2)}")

    results = []

    for ordering_name, ordering in orderings.items():
        print(f"\n=== {ordering_name} ===")
        for k in range(1, 13):
            selected_ids = set(ordering[:k])
            sub_atlas = subset_atlas(full_atlas, selected_ids)
            pack = derive_ffn_compression(sub_atlas, budget)
            student_model = swap_compressed_ffns(teacher, pack, inplace=False)
            metrics = _evaluate_model(teacher, student_model, corpus, batch_size=4)

            ppl = metrics["perplexity"]
            kl = metrics["teacher_student_kl"]
            agreement = metrics["top1_agreement"]
            rel_ppl_inc = (ppl - teacher_ppl) / teacher_ppl

            row = {
                "ordering": ordering_name,
                "k": k,
                "layers_compressed": ordering[:k],
                "perplexity": ppl,
                "relative_ppl_increase": rel_ppl_inc,
                "teacher_student_kl": kl,
                "top1_agreement": agreement,
            }
            results.append(row)
            print(f"  k={k:2d}  layers={ordering[:k]}  ppl={ppl:.3f} ({rel_ppl_inc:+.2%})  kl={kl:.4f}")

            del student_model

    # Save raw results
    output_json = OUTPUT_DIR / "cycle_007_accumulation.json"
    with open(output_json, "w") as f:
        json.dump({"teacher_ppl": teacher_ppl, "results": results}, f, indent=2)
    print(f"\nResults saved to {output_json}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ordering_name in orderings:
        rows = [r for r in results if r["ordering"] == ordering_name]
        ks = [r["k"] for r in rows]
        ppls = [r["perplexity"] for r in rows]
        kls = [r["teacher_student_kl"] for r in rows]
        ax1.plot(ks, ppls, "o-", label=ordering_name, markersize=5)
        ax2.plot(ks, kls, "o-", label=ordering_name, markersize=5)

    ax1.axhline(teacher_ppl, color="gray", linestyle="--", alpha=0.5, label="teacher")
    ax1.set_xlabel("Number of blocks compressed (k)")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("Progressive Multi-Block Accumulation (16x, proxy gpt2)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Number of blocks compressed (k)")
    ax2.set_ylabel("KL divergence (teacher → student)")
    ax2.set_title("KL Divergence vs. Blocks Compressed")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = OUTPUT_DIR / "cycle_007_accumulation.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")

    # ── Evaluate against criteria ──────────────────────────────────
    print("\n=== Criteria evaluation ===")
    # Compare perplexity at k=3,6,9,12 across orderings
    for test_k in [3, 6, 9, 12]:
        easy = next(r for r in results if r["ordering"] == "easiest_first" and r["k"] == test_k)
        hard = next(r for r in results if r["ordering"] == "hardest_first" and r["k"] == test_k)
        rand = next(r for r in results if r["ordering"] == "random" and r["k"] == test_k)

        # ">=25% lower perplexity" means (hard_ppl - easy_ppl) / hard_ppl >= 0.25
        # or equivalently easy_ppl <= 0.75 * hard_ppl
        ppl_reduction_vs_hard = (hard["perplexity"] - easy["perplexity"]) / hard["perplexity"]
        ppl_reduction_vs_rand = (rand["perplexity"] - easy["perplexity"]) / rand["perplexity"]

        print(f"  k={test_k}: easy_ppl={easy['perplexity']:.3f}  hard_ppl={hard['perplexity']:.3f}  "
              f"rand_ppl={rand['perplexity']:.3f}  "
              f"reduction_vs_hard={ppl_reduction_vs_hard:.2%}  "
              f"reduction_vs_rand={ppl_reduction_vs_rand:.2%}")


if __name__ == "__main__":
    main()
