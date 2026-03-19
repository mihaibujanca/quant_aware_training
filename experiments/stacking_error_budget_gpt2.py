"""Cycle 058: Multi-block stacking error budget — GPT-2 FFN.

Tests whether PPL degradation is additive or compounding when stacking
compressed FFN blocks.

Progressive stacking (easiest-first order): k=1 (block 7), k=2 (7+5),
k=3 (7+5+4), k=4 (7+5+4+1).

All at 6x compression with OpenWebText probe/eval.
3 seeds (42, 123, 7), 200 eval sequences.

Primary metric: marginal cost curve convexity (second differences).
  Convex (increasing marginals) → super-additive compounding.
  Concave/flat → sub-additive / additive.

Secondary: ratio R = stacked_delta_k / sum(single_deltas_1..k).
  Only valid when all single deltas > 0; otherwise marked ill-defined.

Residual-stream MSE tracked at each block boundary via forward hooks.
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
TARGET_BLOCKS = [7, 5, 4, 1]   # Progressive stacking order (easiest first)
HIDDEN_FRACTION = 1 / 6         # 6x compression
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
    """Build a CompressedFfnPack for a single block."""
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


def _measure_residual_mse(
    teacher: TeacherBundle,
    student_model: nn.Module,
    probe_tokens: torch.Tensor,
    block_ids: list[int],
) -> dict[int, float]:
    """Measure per-block residual-stream MSE between float teacher and compressed student.

    Hooks the output of each transformer block (after FFN), computes MSE
    averaged over all token positions over the probe batch.
    """
    device = teacher.device
    probe_tokens = probe_tokens.to(device)

    float_hiddens: dict[int, torch.Tensor] = {}
    student_hiddens: dict[int, torch.Tensor] = {}

    def make_hook(store: dict, key: int):
        def hook(module, input, output):
            # GPT-2 block returns a tuple; residual stream is output[0]
            h = output[0] if isinstance(output, tuple) else output
            store[key] = h.detach().float().cpu()
        return hook

    # Register hooks on teacher
    teacher_hooks = []
    for bid in block_ids:
        h = teacher.model.transformer.h[bid].register_forward_hook(
            make_hook(float_hiddens, bid)
        )
        teacher_hooks.append(h)

    with torch.no_grad():
        _ = teacher.model(probe_tokens)
    for h in teacher_hooks:
        h.remove()

    # Register hooks on student
    student_hooks = []
    for bid in block_ids:
        h = student_model.transformer.h[bid].register_forward_hook(
            make_hook(student_hiddens, bid)
        )
        student_hooks.append(h)

    with torch.no_grad():
        _ = student_model(probe_tokens)
    for h in student_hooks:
        h.remove()

    mse_per_block: dict[int, float] = {}
    for bid in block_ids:
        if bid in float_hiddens and bid in student_hiddens:
            diff = float_hiddens[bid] - student_hiddens[bid]
            mse = float(diff.pow(2).mean().item())
        else:
            mse = float("nan")
        mse_per_block[bid] = mse

    return mse_per_block


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

    # Storage: per seed
    # singles[seed][block_id] = ppl
    # stacked[seed][k] = ppl (k = number of blocks stacked, 1..4)
    # mse[seed][k][block_id] = float
    singles_per_seed: dict[int, dict[int, float]] = {s: {} for s in SEEDS}
    stacked_per_seed: dict[int, dict[int, float]] = {s: {} for s in SEEDS}
    mse_per_seed: dict[int, dict[int, dict[int, float]]] = {s: {} for s in SEEDS}

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

        # ---- Single-block baselines: each block independently ----
        print(f"\n  -- Single-block baselines --")
        for block_id in TARGET_BLOCKS:
            _set_seed(seed)
            single_atlas = probe_ffn_geometry(
                teacher,
                owt_corpus,
                layer_ids=(block_id,),
                max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
                batch_size=BATCH_SIZE,
            )
            pack = _derive_single_block_pack(single_atlas, budget, block_id)
            student = swap_compressed_ffns(teacher, pack, inplace=False)
            metrics = _evaluate_model(teacher, student, owt_corpus, batch_size=BATCH_SIZE)
            ppl = metrics["perplexity"]
            singles_per_seed[seed][block_id] = ppl
            delta_pct = 100.0 * (ppl - baseline_ppl) / baseline_ppl
            print(f"  Block {block_id}: PPL={ppl:.3f} ({delta_pct:+.1f}%)")

        # ---- Progressive stacking ----
        print(f"\n  -- Progressive stacking --")
        for k in range(1, len(TARGET_BLOCKS) + 1):
            blocks_k = TARGET_BLOCKS[:k]
            print(f"  k={k}: blocks {blocks_k}")
            _set_seed(seed)

            stacked_atlas = probe_ffn_geometry(
                teacher,
                owt_corpus,
                layer_ids=tuple(blocks_k),
                max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
                batch_size=BATCH_SIZE,
            )
            pack_k = derive_ffn_compression(stacked_atlas, budget)
            student_k = swap_compressed_ffns(teacher, pack_k, inplace=False)
            metrics_k = _evaluate_model(teacher, student_k, owt_corpus, batch_size=BATCH_SIZE)
            ppl_k = metrics_k["perplexity"]
            stacked_per_seed[seed][k] = ppl_k
            delta_pct = 100.0 * (ppl_k - baseline_ppl) / baseline_ppl
            print(f"    Stacked PPL={ppl_k:.3f} ({delta_pct:+.1f}%)")

            # Residual-stream MSE at each block in the stack
            # Use first 4 sequences from eval corpus for probing
            probe_ids = owt_corpus.eval_val[:4]
            mse_dict = _measure_residual_mse(teacher, student_k, probe_ids, blocks_k)
            mse_per_seed[seed][k] = {int(bid): float(v) for bid, v in mse_dict.items()}
            print(f"    Residual MSE: {mse_dict}")

    # ---- Aggregate over seeds ----
    def agg(vals: list[float]) -> tuple[float, float]:
        t = torch.tensor(vals, dtype=torch.float32)
        return float(t.mean().item()), float(t.std().item()) if len(vals) > 1 else 0.0

    # Singles
    singles_agg: dict[int, dict] = {}
    for block_id in TARGET_BLOCKS:
        ppls = [singles_per_seed[s][block_id] for s in SEEDS]
        mean_ppl, std_ppl = agg(ppls)
        deltas = [100.0 * (p - baseline_ppl) / baseline_ppl for p in ppls]
        mean_delta, std_delta = agg(deltas)
        singles_agg[block_id] = {
            "ppl_mean": mean_ppl,
            "ppl_std": std_ppl,
            "delta_pct_mean": mean_delta,
            "delta_pct_std": std_delta,
            "per_seed": {s: singles_per_seed[s][block_id] for s in SEEDS},
        }

    # Stacked
    stacked_agg: dict[int, dict] = {}
    for k in range(1, len(TARGET_BLOCKS) + 1):
        ppls = [stacked_per_seed[s][k] for s in SEEDS]
        mean_ppl, std_ppl = agg(ppls)
        deltas = [100.0 * (p - baseline_ppl) / baseline_ppl for p in ppls]
        mean_delta, std_delta = agg(deltas)
        stacked_agg[k] = {
            "blocks": TARGET_BLOCKS[:k],
            "ppl_mean": mean_ppl,
            "ppl_std": std_ppl,
            "delta_pct_mean": mean_delta,
            "delta_pct_std": std_delta,
            "per_seed": {s: stacked_per_seed[s][k] for s in SEEDS},
        }

    # Marginals: marginal_k = delta_k - delta_{k-1}
    marginals: dict[int, dict] = {}
    for k in range(1, len(TARGET_BLOCKS) + 1):
        per_seed_marginals = []
        for s in SEEDS:
            delta_k = 100.0 * (stacked_per_seed[s][k] - baseline_ppl) / baseline_ppl
            if k == 1:
                delta_prev = 0.0
            else:
                delta_prev = 100.0 * (stacked_per_seed[s][k - 1] - baseline_ppl) / baseline_ppl
            per_seed_marginals.append(delta_k - delta_prev)
        mean_m, std_m = agg(per_seed_marginals)
        marginals[k] = {
            "mean": mean_m,
            "std": std_m,
            "per_seed": {s: per_seed_marginals[i] for i, s in enumerate(SEEDS)},
        }

    # Second differences of marginals (convexity signal)
    second_diffs: list[float] = []
    for k in range(2, len(TARGET_BLOCKS) + 1):
        sd = marginals[k]["mean"] - marginals[k - 1]["mean"]
        second_diffs.append(sd)

    all_positive_second_diffs = all(d > 0 for d in second_diffs)
    mean_second_diff = float(torch.tensor(second_diffs, dtype=torch.float32).mean().item()) if second_diffs else 0.0
    marginal_k1 = marginals[1]["mean"]
    marginal_k4 = marginals[4]["mean"]
    ratio_k4_k1 = (marginal_k4 / marginal_k1) if abs(marginal_k1) > 1e-6 else float("nan")

    # Ratio metric R = stacked_delta_k / sum(single_deltas_1..k)
    ratio_metrics: dict[int, dict] = {}
    for k in range(1, len(TARGET_BLOCKS) + 1):
        blocks_k = TARGET_BLOCKS[:k]
        per_seed_ratios = []
        ill_defined_seeds = []
        for s in SEEDS:
            stacked_delta = 100.0 * (stacked_per_seed[s][k] - baseline_ppl) / baseline_ppl
            single_deltas = [100.0 * (singles_per_seed[s][bid] - baseline_ppl) / baseline_ppl for bid in blocks_k]
            sum_singles = sum(single_deltas)
            if sum_singles <= 0 or any(d <= 0 for d in single_deltas):
                ill_defined_seeds.append(s)
                per_seed_ratios.append(float("nan"))
            else:
                per_seed_ratios.append(stacked_delta / sum_singles)

        valid_ratios = [r for r in per_seed_ratios if not math.isnan(r)]
        mean_r, std_r = agg(valid_ratios) if valid_ratios else (float("nan"), float("nan"))
        ratio_metrics[k] = {
            "mean": mean_r,
            "std": std_r,
            "ill_defined_seeds": ill_defined_seeds,
            "per_seed": {s: per_seed_ratios[i] for i, s in enumerate(SEEDS)},
        }

    # MSE: aggregate over seeds
    mse_agg: dict[int, dict] = {}
    for k in range(1, len(TARGET_BLOCKS) + 1):
        blocks_k = TARGET_BLOCKS[:k]
        mse_agg[k] = {}
        for bid in blocks_k:
            vals = [mse_per_seed[s][k].get(bid, float("nan")) for s in SEEDS]
            valid = [v for v in vals if not math.isnan(v)]
            mean_m, std_m = agg(valid) if valid else (float("nan"), 0.0)
            mse_agg[k][int(bid)] = {"mean": mean_m, "std": std_m}

    # Convexity verdict
    if all_positive_second_diffs and mean_second_diff > 1.0 and ratio_k4_k1 > 2.0:
        convexity_verdict = "CONFIRMED_SUPER_ADDITIVE"
    elif all(d <= 0 for d in second_diffs):
        convexity_verdict = "REFUTED_SUB_ADDITIVE"
    else:
        convexity_verdict = "INCONCLUSIVE"

    results = {
        "baseline_ppl": baseline_ppl,
        "target_blocks": TARGET_BLOCKS,
        "hidden_fraction": HIDDEN_FRACTION,
        "seeds": SEEDS,
        "singles": singles_agg,
        "stacked": stacked_agg,
        "marginals": marginals,
        "second_diffs": second_diffs,
        "all_positive_second_diffs": all_positive_second_diffs,
        "mean_second_diff": mean_second_diff,
        "marginal_k1": marginal_k1,
        "marginal_k4": marginal_k4,
        "ratio_k4_k1": ratio_k4_k1,
        "ratio_metrics": ratio_metrics,
        "mse_agg": mse_agg,
        "convexity_verdict": convexity_verdict,
    }

    return results


def make_plot(results: dict, out_path: Path) -> None:
    blocks = results["target_blocks"]
    k_vals = list(range(1, len(blocks) + 1))
    baseline_ppl = results["baseline_ppl"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GPT-2 FFN Multi-block Stacking Error Budget (6×, OWT)", fontsize=13)

    # ---- Left panel: Marginal PPL cost bars + cumulative residual MSE line ----
    ax = axes[0]
    ax2_right = ax.twinx()

    marginal_means = [results["marginals"][k]["mean"] for k in k_vals]
    marginal_stds = [results["marginals"][k]["std"] for k in k_vals]
    block_labels = [f"k={k}\n+block {blocks[k-1]}" for k in k_vals]

    bar_colors = ["#2196F3", "#FF9800", "#E53935", "#9C27B0"]
    bars = ax.bar(
        k_vals, marginal_means, 0.55, yerr=marginal_stds,
        color=bar_colors, alpha=0.85, capsize=4, label="Marginal PPL cost (pp)"
    )

    # Cumulative residual MSE — average over blocks in each stack
    cumulative_mse = []
    for k in k_vals:
        block_ids_k = blocks[:k]
        mses = [results["mse_agg"][k].get(bid, {}).get("mean", float("nan")) for bid in block_ids_k]
        valid = [m for m in mses if not math.isnan(m)]
        cumulative_mse.append(sum(valid) if valid else float("nan"))

    ax2_right.plot(k_vals, cumulative_mse, "k-o", linewidth=2, markersize=6, label="Cumulative residual MSE")
    ax2_right.set_ylabel("Cumulative residual-stream MSE", color="black")
    ax2_right.tick_params(axis="y", labelcolor="black")

    ax.set_xticks(k_vals)
    ax.set_xticklabels(block_labels)
    ax.set_xlabel("Number of stacked blocks (k)")
    ax.set_ylabel("Marginal PPL cost (percentage points)")
    ax.set_title("Marginal cost per additional block")
    ax.axhline(0, color="gray", lw=0.8, ls="--")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    # ---- Right panel: Cumulative PPL delta vs additive sum-of-singles ----
    ax3 = axes[1]

    stacked_deltas = [results["stacked"][k]["delta_pct_mean"] for k in k_vals]
    stacked_stds = [results["stacked"][k]["delta_pct_std"] for k in k_vals]

    # Additive prediction: cumulative sum of single-block deltas in stacking order
    single_deltas_ordered = [results["singles"][bid]["delta_pct_mean"] for bid in blocks]
    additive_prediction = [sum(single_deltas_ordered[:k]) for k in k_vals]

    ax3.errorbar(
        k_vals, stacked_deltas, yerr=stacked_stds,
        fmt="ro-", linewidth=2, markersize=7, capsize=4,
        label="Actual stacked PPL% change"
    )
    ax3.plot(k_vals, additive_prediction, "b--s", linewidth=2, markersize=6,
             label="Additive prediction (sum of singles)")
    ax3.axhline(0, color="gray", lw=0.8, ls="--")

    ax3.set_xticks(k_vals)
    ax3.set_xticklabels(block_labels)
    ax3.set_xlabel("Number of stacked blocks (k)")
    ax3.set_ylabel("PPL % change vs baseline")
    ax3.set_title(f"Actual vs additive — verdict: {results['convexity_verdict']}")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved plot to {out_path}")


def main() -> None:
    print("Cycle 058: Multi-block stacking error budget — GPT-2 FFN (6×, OWT)")
    results = run_experiment()

    json_path = RESULTS_DIR / "cycle_058_stacking_budget.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {json_path}")

    png_path = RESULTS_DIR / "cycle_058_stacking_budget.png"
    make_plot(results, png_path)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Baseline GPT-2 PPL (OWT): {results['baseline_ppl']:.3f}")

    print("\nSingle-block baselines:")
    for bid in results["target_blocks"]:
        s = results["singles"][bid]
        print(f"  Block {bid}: PPL={s['ppl_mean']:.3f}±{s['ppl_std']:.3f}  ({s['delta_pct_mean']:+.2f}%±{s['delta_pct_std']:.2f}%)")

    print("\nProgressive stacking:")
    print(f"  {'k':<4} {'Blocks':<20} {'PPL mean':<12} {'Delta%':<12} {'Marginal pp':<14}")
    print("-" * 65)
    for k in range(1, len(results["target_blocks"]) + 1):
        st = results["stacked"][k]
        mg = results["marginals"][k]
        print(f"  {k:<4} {str(st['blocks']):<20} {st['ppl_mean']:<12.3f} {st['delta_pct_mean']:+.2f}%{' ':<5} {mg['mean']:+.2f}±{mg['std']:.2f}")

    print(f"\nSecond differences (convexity): {[f'{d:+.2f}' for d in results['second_diffs']]}")
    print(f"All positive: {results['all_positive_second_diffs']}")
    print(f"Mean second diff: {results['mean_second_diff']:+.2f} pp")
    print(f"Marginal k1={results['marginal_k1']:+.2f}, k4={results['marginal_k4']:+.2f}, ratio={results['ratio_k4_k1']:.2f}x")

    print(f"\nRatio R (stacked / sum-of-singles):")
    for k in range(1, len(results["target_blocks"]) + 1):
        rm = results["ratio_metrics"][k]
        if rm["ill_defined_seeds"]:
            print(f"  k={k}: ILL-DEFINED (some single deltas ≤ 0, seeds={rm['ill_defined_seeds']})")
        else:
            print(f"  k={k}: R={rm['mean']:.3f}±{rm['std']:.3f}")

    print(f"\n*** VERDICT: {results['convexity_verdict']} ***")


if __name__ == "__main__":
    main()
