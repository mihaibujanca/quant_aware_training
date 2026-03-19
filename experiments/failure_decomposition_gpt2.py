"""Failure decomposition: channel selection vs lstsq refit vs stacking — GPT-2 FFN.

Cycle 057: Decomposes the +48% PPL gap (cycle 054, stacked k=4@6x) into three
components:
  - Stacking: error compounding across multiple block swaps
  - Channel selection: Shakespeare vs OWT channel sets
  - lstsq refit: whether cross-corpus channel selection matters after refitting

Three variants for blocks [7, 5, 4] at 6x, seeds [42, 123, 7]:
  Variant A: OWT-native single-block (probe+select+refit on OWT, swap 1 block)
  Variant B: Cross-selection refit (Shakespeare channels, OWT lstsq, eval OWT)
  Variant C: Full stacked 3-block (all 3 blocks swapped simultaneously, OWT)
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    CompressedFfnLayer,
    CompressedFfnPack,
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _layer_target_hidden,
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
TARGET_BLOCKS = [7, 5, 4]
HIDDEN_FRACTION = 1 / 6  # 6x compression
SEEDS = [42, 123, 7]
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
EVAL_VAL_SEQS = 200
SEQUENCE_LENGTH = 128
BATCH_SIZE = 4

DATA_DIR = Path(__file__).parent.parent / "data"
OWT_PATH = DATA_DIR / "openwebtext_sample.txt"
SHAKESPEARE_PATH = DATA_DIR / "shakespeare.txt"

RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    import random
    import numpy as np
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
    selected_indices_override: torch.Tensor | None = None,
    lstsq_hidden_override: torch.Tensor | None = None,
    lstsq_output_override: torch.Tensor | None = None,
) -> CompressedFfnPack:
    """Build a CompressedFfnPack for a single block, optionally with overrides."""
    compressed_layers = []
    for layer in atlas.layer_geometries:
        if layer.layer_id != block_id:
            continue
        k = _layer_target_hidden(layer, budget)

        if selected_indices_override is not None:
            selected = selected_indices_override
        else:
            selected = _select_hidden_units(layer, budget, k)

        source_w_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
        source_b_in = _standard_bias(layer.handle.c_fc)
        use_output_bias = _standard_bias(layer.handle.c_proj) is not None

        student_w_in = source_w_in[selected].contiguous()
        student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()

        # Use override hidden/output if provided (cross-corpus refit)
        if lstsq_hidden_override is not None and lstsq_output_override is not None:
            hidden = lstsq_hidden_override.to(dtype=torch.float32)
            outputs = lstsq_output_override.to(dtype=torch.float32)
        else:
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


def compute_jaccard(idx_a: torch.Tensor, idx_b: torch.Tensor) -> float:
    set_a = set(idx_a.tolist())
    set_b = set(idx_b.tolist())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 1.0


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

    results = {
        "baseline_ppl": baseline_ppl,
        "blocks": TARGET_BLOCKS,
        "hidden_fraction": HIDDEN_FRACTION,
        "seeds": SEEDS,
        "jaccard": {},
        "variants": {},
    }

    for block_id in TARGET_BLOCKS:
        print(f"\n{'='*60}")
        print(f"Block {block_id}")
        print(f"{'='*60}")

        block_results_A = []
        block_results_B = []
        block_results_C = []
        jaccard_vals = []

        for seed in SEEDS:
            print(f"\n  Seed {seed}...")
            _set_seed(seed)

            # Build OWT probe corpus for this seed
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

            # Build Shakespeare probe corpus for this seed
            shk_corpus = build_probe_corpus(
                ProbeCorpusSpec(
                    teacher=teacher,
                    text_path=str(SHAKESPEARE_PATH),
                    sequence_length=SEQUENCE_LENGTH,
                    probe_train_sequences=PROBE_TRAIN_SEQS,
                    probe_holdout_sequences=PROBE_TRAIN_SEQS,
                    eval_val_sequences=EVAL_VAL_SEQS,
                    shuffle_seed=seed,
                )
            )

            print(f"    OWT probe tokens: {owt_corpus.token_counts}")
            print(f"    Shakespeare probe tokens: {shk_corpus.token_counts}")

            # --- Atlas construction ---
            layer_ids = (block_id,)

            owt_atlas = probe_ffn_geometry(
                teacher,
                owt_corpus,
                layer_ids=layer_ids,
                max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
                batch_size=BATCH_SIZE,
            )

            shk_atlas = probe_ffn_geometry(
                teacher,
                shk_corpus,
                layer_ids=layer_ids,
                max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
                batch_size=BATCH_SIZE,
            )

            owt_layer = owt_atlas.layer_geometries[0]
            shk_layer = shk_atlas.layer_geometries[0]
            k = _layer_target_hidden(owt_layer, budget)
            print(f"    k={k}, hidden_dim={owt_layer.hidden_dim}")

            owt_selected = _select_hidden_units(owt_layer, budget, k)
            shk_selected = _select_hidden_units(shk_layer, budget, k)
            jac = compute_jaccard(owt_selected, shk_selected)
            jaccard_vals.append(jac)
            print(f"    Jaccard(OWT, Shakespeare) = {jac:.4f}")

            # --- Variant A: OWT-native single-block ---
            pack_A = _derive_single_block_pack(owt_atlas, budget, block_id)
            student_A = swap_compressed_ffns(teacher, pack_A, inplace=False)
            metrics_A = _evaluate_model(teacher, student_A, owt_corpus, batch_size=BATCH_SIZE)
            ppl_A = metrics_A["perplexity"]
            ppl_change_A = 100.0 * (ppl_A - baseline_ppl) / baseline_ppl
            block_results_A.append({"ppl": ppl_A, "ppl_pct_change": ppl_change_A})
            print(f"    Variant A (OWT native, single-block): PPL={ppl_A:.3f} ({ppl_change_A:+.1f}%)")

            # --- Variant B: Shakespeare channels, OWT lstsq refit ---
            # Use Shakespeare-selected channels, but gather hidden activations from OWT probes
            shk_selected_for_refit = shk_selected  # channels chosen by Shakespeare atlas
            owt_hidden_for_refit = owt_layer.hidden_sample[:, shk_selected_for_refit].to(dtype=torch.float32)
            owt_output_for_refit = owt_layer.output_sample.to(dtype=torch.float32)

            pack_B = _derive_single_block_pack(
                owt_atlas,
                budget,
                block_id,
                selected_indices_override=shk_selected_for_refit,
                lstsq_hidden_override=owt_hidden_for_refit,
                lstsq_output_override=owt_output_for_refit,
            )
            student_B = swap_compressed_ffns(teacher, pack_B, inplace=False)
            metrics_B = _evaluate_model(teacher, student_B, owt_corpus, batch_size=BATCH_SIZE)
            ppl_B = metrics_B["perplexity"]
            ppl_change_B = 100.0 * (ppl_B - baseline_ppl) / baseline_ppl
            block_results_B.append({"ppl": ppl_B, "ppl_pct_change": ppl_change_B})
            print(f"    Variant B (Shk channels + OWT refit, single-block): PPL={ppl_B:.3f} ({ppl_change_B:+.1f}%)")

            # --- Variant C: Full stacked 3-block (all blocks at once, OWT) ---
            # Build atlas for all 3 blocks simultaneously
            all_layer_ids = tuple(TARGET_BLOCKS)
            owt_full_atlas = probe_ffn_geometry(
                teacher,
                owt_corpus,
                layer_ids=all_layer_ids,
                max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
                batch_size=BATCH_SIZE,
            )
            pack_C = derive_ffn_compression(owt_full_atlas, budget)
            student_C = swap_compressed_ffns(teacher, pack_C, inplace=False)
            metrics_C = _evaluate_model(teacher, student_C, owt_corpus, batch_size=BATCH_SIZE)
            ppl_C = metrics_C["perplexity"]
            ppl_change_C = 100.0 * (ppl_C - baseline_ppl) / baseline_ppl
            block_results_C.append({"ppl": ppl_C, "ppl_pct_change": ppl_change_C})
            print(f"    Variant C (OWT stacked 3-block): PPL={ppl_C:.3f} ({ppl_change_C:+.1f}%)")

        # Aggregate over seeds
        def summarize(records):
            ppls = [r["ppl"] for r in records]
            changes = [r["ppl_pct_change"] for r in records]
            return {
                "ppl_mean": float(torch.tensor(ppls).mean().item()),
                "ppl_std": float(torch.tensor(ppls).std().item()) if len(ppls) > 1 else 0.0,
                "ppl_pct_change_mean": float(torch.tensor(changes).mean().item()),
                "ppl_pct_change_std": float(torch.tensor(changes).std().item()) if len(changes) > 1 else 0.0,
                "per_seed": records,
            }

        block_key = f"block_{block_id}"
        results["variants"].setdefault("A", {})[block_key] = summarize(block_results_A)
        results["variants"].setdefault("B", {})[block_key] = summarize(block_results_B)
        results["variants"].setdefault("C", {})[block_key] = summarize(block_results_C)
        results["jaccard"][block_key] = {
            "mean": float(torch.tensor(jaccard_vals).mean().item()),
            "std": float(torch.tensor(jaccard_vals).std().item()) if len(jaccard_vals) > 1 else 0.0,
            "per_seed": jaccard_vals,
        }

        print(f"\n  Block {block_id} summary:")
        print(f"    A: {results['variants']['A'][block_key]['ppl_pct_change_mean']:+.1f}% ± {results['variants']['A'][block_key]['ppl_pct_change_std']:.1f}%")
        print(f"    B: {results['variants']['B'][block_key]['ppl_pct_change_mean']:+.1f}% ± {results['variants']['B'][block_key]['ppl_pct_change_std']:.1f}%")
        print(f"    C: {results['variants']['C'][block_key]['ppl_pct_change_mean']:+.1f}% ± {results['variants']['C'][block_key]['ppl_pct_change_std']:.1f}%")
        print(f"    Jaccard: {results['jaccard'][block_key]['mean']:.4f}")

    return results


def make_plot(results: dict, out_path: Path) -> None:
    blocks = results["blocks"]
    block_keys = [f"block_{b}" for b in blocks]
    n_blocks = len(blocks)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GPT-2 FFN Failure Decomposition (6×, blocks 4/5/7)", fontsize=13)

    # Left: PPL % change by variant and block
    ax = axes[0]
    x = range(n_blocks)
    width = 0.25
    offsets = [-width, 0, width]
    colors = ["#2196F3", "#FF9800", "#E53935"]
    labels = ["A: OWT native (single)", "B: Shk select + OWT refit", "C: OWT stacked (3-block)"]
    variant_keys = ["A", "B", "C"]

    for i, (vk, color, label) in enumerate(zip(variant_keys, colors, labels)):
        means = [results["variants"][vk][bk]["ppl_pct_change_mean"] for bk in block_keys]
        stds = [results["variants"][vk][bk]["ppl_pct_change_std"] for bk in block_keys]
        xs = [xi + offsets[i] for xi in x]
        ax.bar(xs, means, width * 0.9, yerr=stds, color=color, alpha=0.85,
               label=label, capsize=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Block {b}" for b in blocks])
    ax.set_ylabel("PPL % change vs baseline")
    ax.set_title("Per-block PPL degradation by variant")
    ax.legend(fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(48.0, color="gray", lw=1.0, ls="--", label="Cycle 054 stacked ref (+48%)")

    # Right: Jaccard similarity
    ax2 = axes[1]
    jac_means = [results["jaccard"][bk]["mean"] for bk in block_keys]
    jac_stds = [results["jaccard"][bk]["std"] for bk in block_keys]
    ax2.bar(list(x), jac_means, 0.5, yerr=jac_stds, color="#9C27B0", alpha=0.8, capsize=4)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"Block {b}" for b in blocks])
    ax2.set_ylabel("Jaccard(OWT, Shakespeare)")
    ax2.set_title("Channel selection overlap across corpora")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, color="gray", lw=0.8, ls="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved plot to {out_path}")


def main() -> None:
    print("Cycle 057: Failure decomposition — channel selection vs lstsq refit vs stacking")
    results = run_experiment()

    json_path = RESULTS_DIR / "cycle_057_decomposition.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {json_path}")

    png_path = RESULTS_DIR / "cycle_057_decomposition.png"
    make_plot(results, png_path)

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Baseline PPL: {results['baseline_ppl']:.3f}")
    print(f"\n{'Block':<8} {'A(OWT-native)':<18} {'B(Shk+OWT-refit)':<22} {'C(3-block stack)':<20} {'Jaccard':<10}")
    print("-" * 80)
    for bk, bid in zip([f"block_{b}" for b in results["blocks"]], results["blocks"]):
        a = results["variants"]["A"][bk]["ppl_pct_change_mean"]
        b = results["variants"]["B"][bk]["ppl_pct_change_mean"]
        c = results["variants"]["C"][bk]["ppl_pct_change_mean"]
        j = results["jaccard"][bk]["mean"]
        print(f"{bid:<8} {a:+.1f}%{' ':12} {b:+.1f}%{' ':15} {c:+.1f}%{' ':13} {j:.3f}")

    # Verdict
    print("\n--- Decomposition analysis ---")
    for bk, bid in zip([f"block_{b}" for b in results["blocks"]], results["blocks"]):
        a_change = results["variants"]["A"][bk]["ppl_pct_change_mean"]
        c_change = results["variants"]["C"][bk]["ppl_pct_change_mean"]
        b_change = results["variants"]["B"][bk]["ppl_pct_change_mean"]
        stacking_overhead = c_change - a_change
        selection_effect = b_change - a_change
        print(f"Block {bid}: stacking overhead={stacking_overhead:+.1f}pp, selection effect (B-A)={selection_effect:+.1f}pp")


if __name__ == "__main__":
    main()
