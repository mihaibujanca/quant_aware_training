"""Probe-corpus robustness of FFN geometry signals (cycle 008, item 13).

Splits Shakespeare probe data into two disjoint halves, probes FFN geometry
independently on each, and compares channel_scores, jacobian_spectrum, and
selected_indices across halves. Proxy-only (gpt2+Shakespeare).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
from scipy.stats import spearmanr

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    FfnGeometryAtlas,
    ProbeCorpus,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
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
    probe_train_sequences=128,  # double, will split
    probe_holdout_sequences=64,
    eval_val_sequences=64,
    dataset_name="proxy_shakespeare",
)
HIDDEN_FRACTION = 0.0625  # 16x compression
OUTPUT_DIR = Path("autoresearch/results")


def split_corpus(full: ProbeCorpus) -> tuple[ProbeCorpus, ProbeCorpus]:
    """Split a corpus into two disjoint halves."""
    n_train = full.probe_train.shape[0] // 2
    n_hold = full.probe_holdout.shape[0] // 2
    n_eval = full.eval_val.shape[0] // 2

    a = replace(full,
                probe_train=full.probe_train[:n_train],
                probe_holdout=full.probe_holdout[:n_hold],
                eval_val=full.eval_val[:n_eval],
                token_counts={
                    "probe_train": int(full.probe_train[:n_train].numel()),
                    "probe_holdout": int(full.probe_holdout[:n_hold].numel()),
                    "eval_val": int(full.eval_val[:n_eval].numel()),
                })
    b = replace(full,
                probe_train=full.probe_train[n_train:2*n_train],
                probe_holdout=full.probe_holdout[n_hold:2*n_hold],
                eval_val=full.eval_val[n_eval:2*n_eval],
                token_counts={
                    "probe_train": int(full.probe_train[n_train:2*n_train].numel()),
                    "probe_holdout": int(full.probe_holdout[n_hold:2*n_hold].numel()),
                    "eval_val": int(full.eval_val[n_eval:2*n_eval].numel()),
                })
    return a, b


def jaccard_overlap(set_a: torch.Tensor, set_b: torch.Tensor) -> float:
    """Jaccard overlap between two 1-D index tensors."""
    a = set(set_a.tolist())
    b = set(set_b.tolist())
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> None:
    print("Loading teacher...")
    teacher = load_nanochat_teacher(TEACHER_SPEC)
    print("Building full corpus...")
    full_corpus = build_probe_corpus(
        ProbeCorpusSpec(teacher=teacher, **CORPUS_SPEC_PARTIAL)
    )
    print(f"Full corpus: train={full_corpus.probe_train.shape}, "
          f"holdout={full_corpus.probe_holdout.shape}, eval={full_corpus.eval_val.shape}")

    corpus_a, corpus_b = split_corpus(full_corpus)
    print(f"Half A: train={corpus_a.probe_train.shape}")
    print(f"Half B: train={corpus_b.probe_train.shape}")

    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        min_hidden=32,
        selection_mode="topk",
        jacobian_samples=8,
    )

    print("\nProbing half A...")
    atlas_a = probe_ffn_geometry(teacher, corpus_a, layer_ids=None,
                                 max_tokens_per_layer=2048, batch_size=4,
                                 jacobian_samples=budget.jacobian_samples)
    print(f"Half A: {len(atlas_a.layer_geometries)} layers probed")

    print("Probing half B...")
    atlas_b = probe_ffn_geometry(teacher, corpus_b, layer_ids=None,
                                 max_tokens_per_layer=2048, batch_size=4,
                                 jacobian_samples=budget.jacobian_samples)
    print(f"Half B: {len(atlas_b.layer_geometries)} layers probed")

    pack_a = derive_ffn_compression(atlas_a, budget)
    pack_b = derive_ffn_compression(atlas_b, budget)

    results = []
    print(f"\n{'layer':>5}  {'score_spearman':>14}  {'jac_cosine':>10}  {'jaccard':>8}  {'k_a':>4}  {'k_b':>4}")
    print("-" * 60)

    layers_a = {lg.layer_id: lg for lg in atlas_a.layer_geometries}
    layers_b = {lg.layer_id: lg for lg in atlas_b.layer_geometries}
    packs_a = {cl.layer_id: cl for cl in pack_a.layers}
    packs_b = {cl.layer_id: cl for cl in pack_b.layers}

    for lid in sorted(layers_a.keys()):
        la = layers_a[lid]
        lb = layers_b[lid]

        # Channel score rank correlation
        scores_a = la.channel_scores.numpy()
        scores_b = lb.channel_scores.numpy()
        rho, _ = spearmanr(scores_a, scores_b)

        # Jacobian spectrum cosine similarity
        ja = la.jacobian_spectrum
        jb = lb.jacobian_spectrum
        jac_cos = float(torch.dot(ja, jb) / (ja.norm() * jb.norm() + 1e-12))

        # Selected indices Jaccard overlap
        sel_a = packs_a[lid].selected_indices
        sel_b = packs_b[lid].selected_indices
        jacc = jaccard_overlap(sel_a, sel_b)

        row = {
            "layer_id": lid,
            "channel_score_spearman": float(rho),
            "jacobian_spectrum_cosine": jac_cos,
            "selected_set_jaccard": jacc,
            "k_a": int(sel_a.shape[0]),
            "k_b": int(sel_b.shape[0]),
        }
        results.append(row)
        print(f"{lid:5d}  {rho:14.4f}  {jac_cos:10.4f}  {jacc:8.4f}  {sel_a.shape[0]:4d}  {sel_b.shape[0]:4d}")

    # ── Aggregate ─────────────────────────────────────────────────
    n_blocks = len(results)
    n_score_pass = sum(1 for r in results if r["channel_score_spearman"] >= 0.8)
    n_jaccard_pass = sum(1 for r in results if r["selected_set_jaccard"] >= 0.5)
    n_both_pass = sum(1 for r in results if r["channel_score_spearman"] >= 0.8 and r["selected_set_jaccard"] >= 0.5)

    mean_spearman = sum(r["channel_score_spearman"] for r in results) / n_blocks
    mean_jaccard = sum(r["selected_set_jaccard"] for r in results) / n_blocks
    mean_jac_cos = sum(r["jacobian_spectrum_cosine"] for r in results) / n_blocks

    # Check early vs late behavior
    early_blocks = [r for r in results if r["layer_id"] < 4]
    late_blocks = [r for r in results if r["layer_id"] >= 8]
    early_mean_spearman = sum(r["channel_score_spearman"] for r in early_blocks) / max(len(early_blocks), 1)
    late_mean_spearman = sum(r["channel_score_spearman"] for r in late_blocks) / max(len(late_blocks), 1)
    early_mean_jaccard = sum(r["selected_set_jaccard"] for r in early_blocks) / max(len(early_blocks), 1)
    late_mean_jaccard = sum(r["selected_set_jaccard"] for r in late_blocks) / max(len(late_blocks), 1)

    summary = {
        "n_blocks": n_blocks,
        "n_score_spearman_ge_0.8": n_score_pass,
        "n_jaccard_ge_0.5": n_jaccard_pass,
        "n_both_pass": n_both_pass,
        "mean_spearman": mean_spearman,
        "mean_jaccard": mean_jaccard,
        "mean_jacobian_cosine": mean_jac_cos,
        "early_mean_spearman": early_mean_spearman,
        "late_mean_spearman": late_mean_spearman,
        "early_mean_jaccard": early_mean_jaccard,
        "late_mean_jaccard": late_mean_jaccard,
    }

    print(f"\n=== Summary ===")
    print(f"Score Spearman >= 0.8: {n_score_pass}/{n_blocks}")
    print(f"Jaccard >= 0.5:        {n_jaccard_pass}/{n_blocks}")
    print(f"Both pass:             {n_both_pass}/{n_blocks} (majority = {n_blocks // 2 + 1})")
    print(f"Mean Spearman:         {mean_spearman:.4f}")
    print(f"Mean Jaccard:          {mean_jaccard:.4f}")
    print(f"Mean Jacobian cosine:  {mean_jac_cos:.4f}")
    print(f"Early (0-3) Spearman:  {early_mean_spearman:.4f}  Jaccard: {early_mean_jaccard:.4f}")
    print(f"Late (8-11) Spearman:  {late_mean_spearman:.4f}  Jaccard: {late_mean_jaccard:.4f}")

    print(f"\n=== Criteria ===")
    majority = n_blocks // 2 + 1
    if n_both_pass >= majority:
        print(f"CONFIRMED: {n_both_pass}/{n_blocks} blocks pass both criteria (>= {majority} needed)")
    elif n_score_pass < majority and n_jaccard_pass < majority:
        print(f"REFUTED: score pass={n_score_pass}, jaccard pass={n_jaccard_pass}, both below majority")
    else:
        print(f"Check inconclusive criterion: early vs late block behavior differs")

    # Save
    output = {"summary": summary, "per_block": results}
    output_path = OUTPUT_DIR / "cycle_008_robustness.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
