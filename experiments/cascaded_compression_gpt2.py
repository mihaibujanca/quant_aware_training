"""Cycle 060: Iterative cascaded compression — GPT-2 FFN.

Tests whether re-probing through partially-compressed blocks before fitting
the next block reduces PPL degradation vs independent (float-activation) baseline.

Independent baseline: compress all 4 blocks using float activations (replicates cycle 058 k=4).
Cascaded: compress blocks one at a time in layer order [1, 4, 5, 7],
  re-probing through the partially-compressed model after each block.

6x compression, OWT probe/eval, 3 seeds (42, 123, 7), 200 eval seqs.

Criterion: cascaded ≤+10% = confirmed, ≥+25% = refuted, 10-25% = inconclusive.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.nn.functional as F

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnLayer,
    CompressedFfnPack,
    CompressedMLP,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _layer_target_hidden,
    _load_compressed_layer,
    _replace_ffn_modules,
    _select_hidden_units,
    _standard_weight,
    _standard_bias,
    build_probe_corpus,
    derive_ffn_compression,
    load_nanochat_teacher,
    probe_ffn_geometry,
    swap_compressed_ffns,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_BLOCKS_LAYERORDER = [1, 4, 5, 7]   # Cascaded compression order (layer order)
TARGET_BLOCKS_STACKING = [7, 5, 4, 1]     # Same blocks, easiest-first order for independent baseline
HIDDEN_FRACTION = 1 / 6                    # 6x compression
SEEDS = [42, 123, 7]
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
EVAL_VAL_SEQS = 200
SEQUENCE_LENGTH = 128
BATCH_SIZE = 4

DATA_DIR = Path(__file__).parent.parent / "data"
OWT_PATH = DATA_DIR / "openwebtext_sample.txt"

RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _param_count(w: torch.Tensor, b: torch.Tensor | None) -> int:
    return w.numel() + (0 if b is None else b.numel())


def _derive_single_block_pack(
    atlas: FfnGeometryAtlas,
    budget: CompressionBudgetSpec,
    block_id: int,
) -> CompressedFfnPack:
    """Build a CompressedFfnPack for a single block from an atlas."""
    compressed_layers = []
    for layer in atlas.layer_geometries:
        if layer.layer_id != block_id:
            continue
        k = _layer_target_hidden(layer, budget)
        selected = _select_hidden_units(layer, budget, k)

        source_w_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
        source_b_in = _standard_bias(layer.handle.c_fc)
        use_output_bias = _standard_bias(layer.handle.c_proj) is not None

        student_w_in = source_w_in[selected].contiguous()
        student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()

        hidden = layer.hidden_sample[:, selected].to(dtype=torch.float32)
        outputs = layer.output_sample.to(dtype=torch.float32)
        student_w_out, student_b_out = _fit_output_projection(hidden, outputs, use_bias=use_output_bias)

        source_param_count = _param_count(source_w_in, source_b_in) + _param_count(
            _standard_weight(layer.handle.c_proj), _standard_bias(layer.handle.c_proj)
        )
        compressed_param_count = _param_count(student_w_in, student_b_in) + _param_count(
            student_w_out, student_b_out
        )
        compressed_layers.append(
            CompressedFfnLayer(
                layer_id=layer.layer_id,
                activation_kind=layer.activation_kind,
                source_hidden_dim=layer.hidden_dim,
                compressed_hidden_dim=int(k),
                c_fc_weight=student_w_in,
                c_fc_bias=student_b_in,
                c_proj_weight=student_w_out,
                c_proj_bias=student_b_out,
                selected_indices=selected,
                source_param_count=source_param_count,
                compressed_param_count=compressed_param_count,
                notes={"selection_mode": budget.selection_mode},
            )
        )
    return CompressedFfnPack(
        method="geometric",
        model_family=atlas.model_family,
        dataset_name=atlas.dataset_name,
        layers=tuple(compressed_layers),
        metadata={"sequence_length": atlas.sequence_length},
    )


def _make_teacher_with_model(teacher: TeacherBundle, model: nn.Module) -> TeacherBundle:
    """Create a new TeacherBundle with a different model (same everything else)."""
    return TeacherBundle(
        source=teacher.source,
        model_family=teacher.model_family,
        model_name=teacher.model_name,
        dataset_name=teacher.dataset_name,
        model=model,
        tokenizer=teacher.tokenizer,
        device=teacher.device,
        max_seq_len=teacher.max_seq_len,
        layer_ids=teacher.layer_ids,
        activation_kind=teacher.activation_kind,
        metadata=dict(teacher.metadata),
    )


def run_independent_baseline(
    teacher: TeacherBundle,
    corpus,
    budget: CompressionBudgetSpec,
    seed: int,
) -> tuple[float, dict[int, float]]:
    """Compress all 4 blocks simultaneously using float activations (replicates cycle 058 k=4)."""
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=tuple(TARGET_BLOCKS_LAYERORDER),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )
    pack = derive_ffn_compression(atlas, budget)
    student = swap_compressed_ffns(teacher, pack, inplace=False)
    metrics = _evaluate_model(teacher, student, corpus, batch_size=BATCH_SIZE)

    # Per-block reconstruction MSE
    block_mse: dict[int, float] = {}
    for layer_geo in atlas.layer_geometries:
        # Find compressed layer
        comp = next((c for c in pack.layers if c.layer_id == layer_geo.layer_id), None)
        if comp is None:
            continue
        mlp = CompressedMLP(
            input_dim=comp.c_fc_weight.shape[1],
            hidden_dim=comp.compressed_hidden_dim,
            output_dim=comp.c_proj_weight.shape[0],
            activation_kind=comp.activation_kind,
            c_fc_bias=comp.c_fc_bias is not None,
            c_proj_bias=comp.c_proj_bias is not None,
        )
        _load_compressed_layer(mlp, comp)
        with torch.no_grad():
            pred = mlp(layer_geo.input_sample)
            mse = float(F.mse_loss(pred, layer_geo.output_sample).item())
        block_mse[layer_geo.layer_id] = mse

    return metrics["perplexity"], block_mse


def run_cascaded(
    teacher: TeacherBundle,
    corpus,
    budget: CompressionBudgetSpec,
    seed: int,
) -> tuple[float, dict[int, float]]:
    """Compress blocks one at a time in layer order, re-probing through partially-compressed model."""
    _set_seed(seed)

    # Start from a fresh copy of the teacher model
    working_model = copy.deepcopy(teacher.model)
    working_model.eval()

    compressed_layers_all: list[CompressedFfnLayer] = []
    block_mse: dict[int, float] = {}

    for block_id in TARGET_BLOCKS_LAYERORDER:
        print(f"    Cascaded: probing block {block_id} (working_model has blocks "
              f"{[c.layer_id for c in compressed_layers_all]} compressed)...")

        # Create a temporary teacher bundle pointing to the partially-compressed model
        working_teacher = _make_teacher_with_model(teacher, working_model)

        # Probe this block using the partially-compressed model
        _set_seed(seed)
        single_atlas = probe_ffn_geometry(
            working_teacher,
            corpus,
            layer_ids=(block_id,),
            max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
            batch_size=BATCH_SIZE,
        )

        # Derive compression for this single block
        single_pack = _derive_single_block_pack(single_atlas, budget, block_id)

        # Apply compression in-place to working_model
        _replace_ffn_modules(working_model, teacher.activation_kind, single_pack)
        working_model.eval()

        # Collect layer for final pack
        for layer in single_pack.layers:
            compressed_layers_all.append(layer)

        # Per-block reconstruction MSE (using the distorted inputs from working_model)
        layer_geo = single_atlas.layer_geometries[0]
        comp = single_pack.layers[0]
        mlp = CompressedMLP(
            input_dim=comp.c_fc_weight.shape[1],
            hidden_dim=comp.compressed_hidden_dim,
            output_dim=comp.c_proj_weight.shape[0],
            activation_kind=comp.activation_kind,
            c_fc_bias=comp.c_fc_bias is not None,
            c_proj_bias=comp.c_proj_bias is not None,
        )
        _load_compressed_layer(mlp, comp)
        with torch.no_grad():
            pred = mlp(layer_geo.input_sample)
            mse = float(F.mse_loss(pred, layer_geo.output_sample).item())
        block_mse[block_id] = mse

    # Evaluate the fully-compressed working_model
    # working_model already has all 4 blocks compressed
    metrics = _evaluate_model(teacher, working_model, corpus, batch_size=BATCH_SIZE)
    return metrics["perplexity"], block_mse


def run_experiment() -> dict:
    print("Loading GPT-2 teacher...")
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(
            source="hf_pretrained",
            hf_model_name="gpt2",
            device="cpu",
        )
    )
    print(f"Teacher loaded: {teacher.model_name}, device={teacher.device}")

    # Evaluate baseline teacher PPL
    print("Building OWT eval corpus for baseline PPL...")
    owt_eval_corpus = build_probe_corpus(
        ProbeCorpusSpec(
            teacher=teacher,
            text_path=str(OWT_PATH),
            sequence_length=SEQUENCE_LENGTH,
            probe_train_sequences=PROBE_TRAIN_SEQS,
            probe_holdout_sequences=PROBE_TRAIN_SEQS,
            eval_val_sequences=EVAL_VAL_SEQS,
            shuffle_seed=42,
        )
    )
    baseline_metrics = _evaluate_model(teacher, teacher.model, owt_eval_corpus, batch_size=BATCH_SIZE)
    baseline_ppl = baseline_metrics["perplexity"]
    print(f"Baseline GPT-2 PPL (OWT): {baseline_ppl:.3f}")

    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        min_hidden=32,
        selection_mode="topk",
    )

    # Per-seed results
    indep_ppls: list[float] = []
    casc_ppls: list[float] = []
    indep_mse_per_seed: list[dict[int, float]] = []
    casc_mse_per_seed: list[dict[int, float]] = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        _set_seed(seed)

        owt_corpus = build_probe_corpus(
            ProbeCorpusSpec(
                teacher=teacher,
                text_path=str(OWT_PATH),
                sequence_length=SEQUENCE_LENGTH,
                probe_train_sequences=PROBE_TRAIN_SEQS,
                probe_holdout_sequences=PROBE_TRAIN_SEQS,
                eval_val_sequences=EVAL_VAL_SEQS,
                shuffle_seed=seed,
            )
        )
        print(f"  OWT probe tokens: {owt_corpus.token_counts}")

        # ---- Independent baseline ----
        print(f"\n  -- Independent baseline (float activations) --")
        _set_seed(seed)
        indep_ppl, indep_mse = run_independent_baseline(teacher, owt_corpus, budget, seed)
        indep_delta = 100.0 * (indep_ppl - baseline_ppl) / baseline_ppl
        print(f"  Independent PPL={indep_ppl:.3f} ({indep_delta:+.1f}%)")
        print(f"  Per-block MSE: {indep_mse}")
        indep_ppls.append(indep_ppl)
        indep_mse_per_seed.append(indep_mse)

        # ---- Cascaded variant ----
        print(f"\n  -- Cascaded (re-probe through compressed blocks) --")
        _set_seed(seed)
        casc_ppl, casc_mse = run_cascaded(teacher, owt_corpus, budget, seed)
        casc_delta = 100.0 * (casc_ppl - baseline_ppl) / baseline_ppl
        print(f"  Cascaded PPL={casc_ppl:.3f} ({casc_delta:+.1f}%)")
        print(f"  Per-block MSE: {casc_mse}")
        casc_ppls.append(casc_ppl)
        casc_mse_per_seed.append(casc_mse)

    # ---- Aggregate ----
    def agg(vals: list[float]) -> tuple[float, float]:
        t = torch.tensor(vals, dtype=torch.float32)
        return float(t.mean().item()), float(t.std().item()) if len(vals) > 1 else 0.0

    indep_mean, indep_std = agg(indep_ppls)
    casc_mean, casc_std = agg(casc_ppls)

    indep_deltas = [100.0 * (p - baseline_ppl) / baseline_ppl for p in indep_ppls]
    casc_deltas = [100.0 * (p - baseline_ppl) / baseline_ppl for p in casc_ppls]

    indep_delta_mean, indep_delta_std = agg(indep_deltas)
    casc_delta_mean, casc_delta_std = agg(casc_deltas)

    # Per-block MSE means across seeds
    def mean_mse_by_block(mse_list: list[dict[int, float]]) -> dict[int, float]:
        result: dict[int, float] = {}
        all_blocks = set()
        for d in mse_list:
            all_blocks |= set(d.keys())
        for bid in sorted(all_blocks):
            vals = [d[bid] for d in mse_list if bid in d]
            result[bid] = float(torch.tensor(vals, dtype=torch.float32).mean().item()) if vals else float("nan")
        return result

    # Verdict
    if casc_delta_mean <= 10.0:
        verdict = "CONFIRMED"
    elif casc_delta_mean >= 25.0:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    results = {
        "baseline_ppl": baseline_ppl,
        "target_blocks": TARGET_BLOCKS_LAYERORDER,
        "hidden_fraction": HIDDEN_FRACTION,
        "seeds": SEEDS,
        "ppl_float": baseline_ppl,
        "ppl_independent_mean": indep_mean,
        "ppl_independent_std": indep_std,
        "ppl_cascaded_mean": casc_mean,
        "ppl_cascaded_std": casc_std,
        "pct_degrad_independent_mean": indep_delta_mean,
        "pct_degrad_independent_std": indep_delta_std,
        "pct_degrad_cascaded_mean": casc_delta_mean,
        "pct_degrad_cascaded_std": casc_delta_std,
        "ppl_independent_per_seed": {s: v for s, v in zip(SEEDS, indep_ppls)},
        "ppl_cascaded_per_seed": {s: v for s, v in zip(SEEDS, casc_ppls)},
        "pct_degrad_independent_per_seed": {s: v for s, v in zip(SEEDS, indep_deltas)},
        "pct_degrad_cascaded_per_seed": {s: v for s, v in zip(SEEDS, casc_deltas)},
        "mse_independent_by_block": {int(k): v for k, v in mean_mse_by_block(indep_mse_per_seed).items()},
        "mse_cascaded_by_block": {int(k): v for k, v in mean_mse_by_block(casc_mse_per_seed).items()},
        "verdict": verdict,
    }
    return results


def make_plot(results: dict, out_path: Path) -> None:
    blocks = results["target_blocks"]
    baseline_ppl = results["ppl_float"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GPT-2 FFN Cascaded vs Independent Compression (6×, OWT)", fontsize=13)

    # ---- Left panel: PPL bar chart ----
    ax = axes[0]
    labels = ["Float baseline", "Independent", "Cascaded"]
    ppls = [
        baseline_ppl,
        results["ppl_independent_mean"],
        results["ppl_cascaded_mean"],
    ]
    stds = [0.0, results["ppl_independent_std"], results["ppl_cascaded_std"]]
    colors = ["#4CAF50", "#2196F3", "#FF9800"]
    bars = ax.bar(labels, ppls, 0.5, yerr=stds, color=colors, alpha=0.85, capsize=5)
    ax.set_ylabel("Perplexity (lower = better)")
    ax.set_title("PPL: Float vs Independent vs Cascaded")
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2.0, ppl + 0.3,
                f"{ppl:.2f}", ha="center", va="bottom", fontsize=9)

    # ---- Right panel: per-block reconstruction MSE ----
    ax2 = axes[1]
    x = list(sorted(set(results["mse_independent_by_block"].keys()) | set(results["mse_cascaded_by_block"].keys())))
    x_ticks = list(range(len(x)))
    indep_mses = [results["mse_independent_by_block"].get(b, float("nan")) for b in x]
    casc_mses = [results["mse_cascaded_by_block"].get(b, float("nan")) for b in x]

    w = 0.35
    ax2.bar([t - w / 2 for t in x_ticks], indep_mses, w, label="Independent", color="#2196F3", alpha=0.85)
    ax2.bar([t + w / 2 for t in x_ticks], casc_mses, w, label="Cascaded", color="#FF9800", alpha=0.85)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f"Block {b}" for b in x])
    ax2.set_ylabel("Reconstruction MSE (per block)")
    ax2.set_title("Per-block FFN reconstruction MSE")
    ax2.legend(fontsize=9)

    # Annotate verdict
    verdict = results["verdict"]
    indep_d = results["pct_degrad_independent_mean"]
    casc_d = results["pct_degrad_cascaded_mean"]
    fig.text(0.5, 0.01,
             f"Independent: {indep_d:+.1f}%  |  Cascaded: {casc_d:+.1f}%  |  Verdict: {verdict}",
             ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved plot to {out_path}")


def main() -> None:
    print("Cycle 060: Iterative cascaded compression — GPT-2 FFN (6×, OWT)")
    results = run_experiment()

    json_path = RESULTS_DIR / "cycle_060_cascaded.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {json_path}")

    png_path = RESULTS_DIR / "cycle_060_cascaded.png"
    make_plot(results, png_path)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — Cycle 060: Cascaded vs Independent Compression")
    print("=" * 60)
    print(f"Baseline GPT-2 PPL (OWT): {results['ppl_float']:.3f}")
    print(f"Independent baseline:")
    print(f"  PPL={results['ppl_independent_mean']:.3f}±{results['ppl_independent_std']:.3f}  "
          f"({results['pct_degrad_independent_mean']:+.2f}%±{results['pct_degrad_independent_std']:.2f}%)")
    print(f"  Per-seed: {results['ppl_independent_per_seed']}")
    print(f"Cascaded:")
    print(f"  PPL={results['ppl_cascaded_mean']:.3f}±{results['ppl_cascaded_std']:.3f}  "
          f"({results['pct_degrad_cascaded_mean']:+.2f}%±{results['pct_degrad_cascaded_std']:.2f}%)")
    print(f"  Per-seed: {results['ppl_cascaded_per_seed']}")
    print(f"\nPer-block reconstruction MSE:")
    print(f"  Independent: {results['mse_independent_by_block']}")
    print(f"  Cascaded:    {results['mse_cascaded_by_block']}")
    print(f"\n*** VERDICT: {results['verdict']} ***")
    print(f"  (Cascaded {results['pct_degrad_cascaded_mean']:+.1f}% vs criterion ≤+10% confirmed, ≥+25% refuted)")


if __name__ == "__main__":
    main()
