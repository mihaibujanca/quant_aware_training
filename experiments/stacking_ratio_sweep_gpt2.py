"""Cycle 062: Reduced compression ratio sweep for k=4 stacking — GPT-2 FFN.

Tests k=4 stacking (blocks [7,5,4,1]) at per-block ratios {2x, 3x, 4x, 6x}.
Pure analytic (topk + lstsq), NO gradient polish.
6x is replication baseline from cycle 058 (~+30.2% PPL).

Critical fix from cycle 056: 8192 probe tokens (vs 4096) to avoid underdetermined
lstsq at 2x (8192 tokens × 1537 columns = 5.3x overdetermination).

Criterion: k=4@3x <= +5% PPL degradation → CONFIRMED
           k=4@3x >= +20% PPL degradation → REFUTED
           +5-20% → INCONCLUSIVE
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
import torch.nn.functional as F

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
    _iter_lm_batches,
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
TARGET_BLOCKS = [7, 5, 4, 1]   # Easiest-first stacking order (from cycle 058)
RATIOS = [2, 3, 4, 6]           # 6x is calibration baseline
SEEDS = [42, 123, 7]
MAX_TOKENS_PER_LAYER = 8192     # Critical fix: 8192 (not 4096) to avoid underdetermined lstsq at 2x
PROBE_TRAIN_SEQS = 1            # Minimum to avoid _token_windows ValueError; probe_train not used
PROBE_HOLDOUT_SEQS = 200        # Provides 25600 tokens, well above 8192 needed
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
    print("Building OWT corpus for baseline PPL...")
    owt_eval_corpus = build_probe_corpus(
        ProbeCorpusSpec(
            teacher=teacher,
            text_path=str(OWT_PATH),
            sequence_length=SEQUENCE_LENGTH,
            probe_train_sequences=PROBE_TRAIN_SEQS,
            probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
            eval_val_sequences=EVAL_VAL_SEQS,
            shuffle_seed=42,
        )
    )
    baseline_metrics = _evaluate_model(teacher, teacher.model, owt_eval_corpus, batch_size=BATCH_SIZE)
    baseline_ppl = baseline_metrics["perplexity"]
    print(f"Baseline GPT-2 PPL (OWT): {baseline_ppl:.3f}")

    # Results storage: ratio_ppls[seed][ratio] = ppl
    ratio_ppls: dict[int, dict[int, float]] = {s: {} for s in SEEDS}

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
                probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
                eval_val_sequences=EVAL_VAL_SEQS,
                shuffle_seed=seed,
            )
        )
        print(f"  OWT corpus token counts: {owt_corpus.token_counts}")

        # Hoist atlas computation outside ratio loop: atlas depends only on corpus + teacher
        print(f"\n  -- Probing FFN geometry for blocks {TARGET_BLOCKS} --")
        _set_seed(seed)
        atlas = probe_ffn_geometry(
            teacher,
            owt_corpus,
            layer_ids=tuple(TARGET_BLOCKS),
            max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
            batch_size=BATCH_SIZE,
        )
        print(f"  Atlas probed {len(atlas.layer_geometries)} layers")

        for ratio in RATIOS:
            hidden_fraction = 1.0 / ratio
            print(f"\n  -- Ratio = {ratio}x (hidden_fraction={hidden_fraction:.4f}) --")
            _set_seed(seed)

            budget = CompressionBudgetSpec(
                hidden_fraction=hidden_fraction,
                min_hidden=32,
                selection_mode="topk",
            )

            pack_all = derive_ffn_compression(atlas, budget)
            student = swap_compressed_ffns(teacher, pack_all, inplace=False)
            student.eval()

            metrics = _evaluate_model(teacher, student, owt_corpus, batch_size=BATCH_SIZE)
            ppl = metrics["perplexity"]
            ratio_ppls[seed][ratio] = ppl
            delta_pct = 100.0 * (ppl - baseline_ppl) / baseline_ppl
            print(f"  ratio={ratio}x: PPL={ppl:.3f} ({delta_pct:+.1f}%)")

    # ---- Aggregate over seeds ----
    def agg(vals: list[float]) -> tuple[float, float]:
        t = torch.tensor(vals, dtype=torch.float32)
        return float(t.mean().item()), float(t.std().item()) if len(vals) > 1 else 0.0

    ratio_agg: dict[int, dict] = {}
    for ratio in RATIOS:
        ppls = [ratio_ppls[s][ratio] for s in SEEDS]
        mean_ppl, std_ppl = agg(ppls)
        deltas = [100.0 * (p - baseline_ppl) / baseline_ppl for p in ppls]
        mean_delta, std_delta = agg(deltas)
        ratio_agg[ratio] = {
            "ppl_mean": mean_ppl,
            "ppl_std": std_ppl,
            "delta_pct_mean": mean_delta,
            "delta_pct_std": std_delta,
            "per_seed": {s: ratio_ppls[s][ratio] for s in SEEDS},
        }

    # Verdict based on 3x result
    delta_3x = ratio_agg[3]["delta_pct_mean"]
    if delta_3x <= 5.0:
        verdict = "CONFIRMED"
    elif delta_3x >= 20.0:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    # 6x calibration check vs cycle 058 (~75.06 mean PPL, +30.2%)
    cycle058_baseline = 75.06  # cycle 058 reported mean 6x PPL
    delta_6x = ratio_agg[6]["delta_pct_mean"]
    calibration_deviation = abs(delta_6x - 30.2)
    calibration_ok = calibration_deviation <= 3.0

    results = {
        "baseline_ppl": baseline_ppl,
        "target_blocks": TARGET_BLOCKS,
        "ratios": RATIOS,
        "seeds": SEEDS,
        "max_tokens_per_layer": MAX_TOKENS_PER_LAYER,
        "ratio_agg": ratio_agg,
        "verdict": verdict,
        "verdict_basis": f"k=4@3x delta={delta_3x:+.2f}%",
        "calibration": {
            "cycle058_ref_6x_delta_pct": 30.2,
            "measured_6x_delta_pct": delta_6x,
            "deviation_pp": calibration_deviation,
            "calibration_ok": calibration_ok,
        },
    }

    return results


def make_plot(results: dict, out_path: Path) -> None:
    ratios = results["ratios"]
    baseline_ppl = results["baseline_ppl"]

    delta_means = [results["ratio_agg"][r]["delta_pct_mean"] for r in ratios]
    delta_stds = [results["ratio_agg"][r]["delta_pct_std"] for r in ratios]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"k=4 Stacking Ratio Sweep — GPT-2 FFN (OWT)\n"
        f"Baseline PPL={baseline_ppl:.3f}  Verdict: {results['verdict']}",
        fontsize=12,
    )

    colors = ["#4CAF50", "#2196F3", "#FF9800", "#E53935"]
    ax.bar(range(len(ratios)), delta_means, 0.55, yerr=delta_stds,
           color=colors[:len(ratios)], alpha=0.85, capsize=5)
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"{r}×" for r in ratios])
    ax.set_xlabel("Compression ratio (per block)")
    ax.set_ylabel("PPL % change vs baseline")
    ax.set_title("PPL degradation by compression ratio (no polish)")

    # Threshold lines
    ax.axhline(5.0, color="green", lw=1.5, ls="--", label="+5% (confirmed threshold)")
    ax.axhline(20.0, color="orange", lw=1.5, ls="--", label="+20% (refuted threshold)")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.legend(fontsize=9)

    for i, (m, s) in enumerate(zip(delta_means, delta_stds)):
        ax.text(i, m + s + 0.5, f"{m:+.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved plot to {out_path}")


def main() -> None:
    print("Cycle 062: k=4 Stacking Ratio Sweep — GPT-2 FFN (OWT)")
    results = run_experiment()

    json_path = RESULTS_DIR / "cycle_062_stacking_ratio_sweep.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {json_path}")

    png_path = RESULTS_DIR / "cycle_062_stacking_ratio_sweep.png"
    make_plot(results, png_path)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Baseline GPT-2 PPL (OWT): {results['baseline_ppl']:.3f}")
    print(f"Target blocks (k=4): {results['target_blocks']}")
    print(f"Max probe tokens per layer: {results['max_tokens_per_layer']}")

    print(f"\n{'Ratio':<10} {'PPL mean':<12} {'Delta%':<12} {'Std%':<10}")
    print("-" * 50)
    for r in results["ratios"]:
        d = results["ratio_agg"][r]
        print(f"{r}x{' ':<7} {d['ppl_mean']:<12.3f} {d['delta_pct_mean']:+.2f}%{' ':<5} {d['delta_pct_std']:.2f}%")

    cal = results["calibration"]
    print(f"\n6x calibration vs cycle 058: measured={cal['measured_6x_delta_pct']:+.1f}% "
          f"ref=+30.2% deviation={cal['deviation_pp']:.1f}pp "
          f"({'OK' if cal['calibration_ok'] else 'EXCEEDED — CHECK'})")

    print(f"\n*** VERDICT: {results['verdict']} ***")
    print(f"Basis: {results['verdict_basis']}")
    criterion_met = results["verdict"] == "CONFIRMED"
    print(f"Criterion (k=4@3x <= +5%): {'MET' if criterion_met else 'NOT MET'}")


if __name__ == "__main__":
    main()
