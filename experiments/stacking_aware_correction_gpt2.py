"""Cycle 063: Stacking-aware boundary correction — GPT-2 FFN.

Two variants:
  A. Naive corrections: each C_i fitted on float activations (replicates cycle 043 on OWT).
  B. Cascaded corrections: C_1 on float, C_2 on (compressed+C_1) output, etc.

Each correction is a full affine map C_i: h -> A_i·h + b_i fitted via lstsq.
Uses all probe data (25K+ tokens) with 80/20 train/held-out split.
Logs per-correction held-out MSE to check for overfitting (flags at 1.5x ratio).

k=4@6x on blocks [7, 5, 4, 1], OWT probe/eval, 3 seeds, 200 eval seqs.
Baseline (uncorrected stacked): +30.2% (cycle 058)
Criterion: cascaded ≤+12% = CONFIRMED, worse than uncorrected = REFUTED
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
    CompressedFfnPack,
    ProbeCorpusSpec,
    TeacherBundle,
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
TARGET_BLOCKS = [1, 4, 5, 7]          # blocks to compress (used in order)
HIDDEN_FRACTION = 1 / 6               # 6x compression
SEEDS = [42, 123, 7]
PROBE_TRAIN_SEQS = 200                 # 200×128 = 25600 tokens
EVAL_VAL_SEQS = 200
SEQUENCE_LENGTH = 128
BATCH_SIZE = 4
CORRECTION_HELD_OUT_FRAC = 0.2        # 80/20 split

DATA_DIR = Path(__file__).parent.parent / "data"
OWT_PATH = DATA_DIR / "openwebtext_sample.txt"

RESULTS_DIR = Path(__file__).parent.parent / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OVERFITTING_THRESHOLD = 1.5           # held_out_mse > 1.5 * fit_mse → flag


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_hidden_states(model: nn.Module, input_ids: torch.Tensor, hook_layer_idx: int, batch_size: int = 4) -> torch.Tensor:
    """Collect activations entering transformer block hook_layer_idx (pre-block)."""
    collected = []

    def hook_fn(module, args):
        collected.append(args[0].detach().clone())

    handle = model.transformer.h[hook_layer_idx].register_forward_pre_hook(hook_fn)
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            model(input_ids[i:i + batch_size])
    handle.remove()
    return torch.cat(collected, dim=0)  # (N_seqs, seq_len, d_model)


def fit_correction_with_held_out(
    h_float: torch.Tensor,
    h_comp: torch.Tensor,
    held_out_frac: float = 0.2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit affine correction h_comp -> h_float with 80/20 split.

    Returns:
        W: (d, d) weight matrix
        b: (d,) bias
        diag: dict with fit_mse, held_out_mse, n_fit, n_held_out, overdetermination_ratio, overfitting_flag
    """
    h_f = h_float.reshape(-1, h_float.shape[-1]).float()  # (N, d)
    h_c = h_comp.reshape(-1, h_comp.shape[-1]).float()    # (N, d)

    N, d = h_f.shape

    # 80/20 split
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(N, generator=rng)
    n_held_out = max(1, int(N * held_out_frac))
    n_fit = N - n_held_out

    fit_idx = perm[:n_fit]
    held_idx = perm[n_fit:]

    h_f_fit = h_f[fit_idx]
    h_c_fit = h_c[fit_idx]
    h_f_held = h_f[held_idx]
    h_c_held = h_c[held_idx]

    # Augment with bias column
    H_fit = torch.cat([h_c_fit, torch.ones(n_fit, 1)], dim=1)  # (n_fit, d+1)
    sol = torch.linalg.lstsq(H_fit, h_f_fit).solution         # (d+1, d)
    W = sol[:d].T    # (d, d)
    b = sol[d]       # (d,)

    # Compute MSEs
    h_pred_fit = h_c_fit @ W.T + b
    fit_mse = float(F.mse_loss(h_pred_fit, h_f_fit).item())

    h_pred_held = h_c_held @ W.T + b
    held_out_mse = float(F.mse_loss(h_pred_held, h_f_held).item())

    # Number of parameters per output dim: d+1 (one row of W + one bias element)
    # overdetermination = n_fit / (d + 1)
    overdetermination = n_fit / (d + 1)

    diag = {
        "fit_mse": fit_mse,
        "held_out_mse": held_out_mse,
        "n_fit": n_fit,
        "n_held_out": n_held_out,
        "overdetermination_ratio": overdetermination,
        "overfitting_flag": held_out_mse > OVERFITTING_THRESHOLD * fit_mse,
    }
    return W, b, diag


def make_correction_hook(W: torch.Tensor, b: torch.Tensor):
    """Returns a forward_pre_hook that applies h -> h @ W.T + b."""
    def hook_fn(module, args):
        h = args[0]
        corrected = h @ W.T + b
        return (corrected,) + args[1:]
    return hook_fn


def run_naive_corrections(
    teacher: TeacherBundle,
    comp_model: nn.Module,
    probe_ids: torch.Tensor,
    target_blocks: list[int],
    seed: int,
) -> tuple[list[dict], list]:
    """Fit each correction using float activations (naive — does NOT account for stacking).

    Returns per-correction diagnostics and installed hook handles.
    """
    sorted_blocks = sorted(target_blocks)
    all_diags = []
    hooks = []

    for block_id in sorted_blocks:
        correction_layer = block_id + 1
        if correction_layer >= 12:
            continue

        h_float = collect_hidden_states(teacher.model, probe_ids, correction_layer, BATCH_SIZE)
        h_comp = collect_hidden_states(comp_model, probe_ids, correction_layer, BATCH_SIZE)

        W, b, diag = fit_correction_with_held_out(h_float, h_comp, CORRECTION_HELD_OUT_FRAC, seed)
        diag["block_id"] = block_id
        diag["correction_layer"] = correction_layer
        diag["variant"] = "naive"

        hook = comp_model.transformer.h[correction_layer].register_forward_pre_hook(
            make_correction_hook(W, b)
        )
        hooks.append(hook)

        flag_str = " ** OVERFITTING **" if diag["overfitting_flag"] else ""
        print(f"    Naive block {block_id} -> layer {correction_layer}: "
              f"fit_mse={diag['fit_mse']:.5f}, held_mse={diag['held_out_mse']:.5f}, "
              f"overdet={diag['overdetermination_ratio']:.1f}{flag_str}")
        all_diags.append(diag)

    return all_diags, hooks


def run_cascaded_corrections(
    teacher: TeacherBundle,
    comp_model: nn.Module,
    probe_ids: torch.Tensor,
    target_blocks: list[int],
    seed: int,
) -> tuple[list[dict], list]:
    """Fit each correction on the actual comp_model (with prior corrections installed).

    C_1: fit on (float_1, comp_1)
    C_2: fit on (float_2, comp_with_C1_2)
    etc.
    """
    sorted_blocks = sorted(target_blocks)
    all_diags = []
    hooks = []

    for block_id in sorted_blocks:
        correction_layer = block_id + 1
        if correction_layer >= 12:
            continue

        # Float reference always from teacher
        h_float = collect_hidden_states(teacher.model, probe_ids, correction_layer, BATCH_SIZE)
        # Compressed reference from comp_model with prior corrections already installed
        h_comp = collect_hidden_states(comp_model, probe_ids, correction_layer, BATCH_SIZE)

        W, b, diag = fit_correction_with_held_out(h_float, h_comp, CORRECTION_HELD_OUT_FRAC, seed)
        diag["block_id"] = block_id
        diag["correction_layer"] = correction_layer
        diag["variant"] = "cascaded"

        hook = comp_model.transformer.h[correction_layer].register_forward_pre_hook(
            make_correction_hook(W, b)
        )
        hooks.append(hook)

        flag_str = " ** OVERFITTING **" if diag["overfitting_flag"] else ""
        print(f"    Cascaded block {block_id} -> layer {correction_layer}: "
              f"fit_mse={diag['fit_mse']:.5f}, held_mse={diag['held_out_mse']:.5f}, "
              f"overdet={diag['overdetermination_ratio']:.1f}{flag_str}")
        all_diags.append(diag)

    return all_diags, hooks


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

    # Float baseline PPL
    print("Building OWT eval corpus for baseline PPL...")
    owt_baseline_corpus = build_probe_corpus(
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
    baseline_metrics = _evaluate_model(teacher, teacher.model, owt_baseline_corpus, batch_size=BATCH_SIZE)
    baseline_ppl = baseline_metrics["perplexity"]
    print(f"Baseline GPT-2 PPL (OWT): {baseline_ppl:.3f}")

    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        min_hidden=32,
        selection_mode="topk",
    )

    # Per-seed storage
    uncorr_ppls: list[float] = []
    naive_ppls: list[float] = []
    casc_ppls: list[float] = []
    naive_diags_all: list[list[dict]] = []
    casc_diags_all: list[list[dict]] = []

    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"SEED {seed}")
        print(f"{'=' * 60}")
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

        # Step 1: Probe float geometry and compress all 4 blocks
        print(f"\n  -- Probing float geometry for blocks {TARGET_BLOCKS} --")
        _set_seed(seed)
        atlas = probe_ffn_geometry(
            teacher,
            owt_corpus,
            layer_ids=tuple(TARGET_BLOCKS),
            max_tokens_per_layer=4096,
            batch_size=BATCH_SIZE,
        )
        pack = derive_ffn_compression(atlas, budget)

        # Step 2: Eval uncorrected stacked PPL
        comp_model_uncorr = swap_compressed_ffns(teacher, pack, inplace=False)
        uncorr_metrics = _evaluate_model(teacher, comp_model_uncorr, owt_corpus, batch_size=BATCH_SIZE)
        uncorr_ppl = uncorr_metrics["perplexity"]
        uncorr_delta = 100.0 * (uncorr_ppl - baseline_ppl) / baseline_ppl
        print(f"\n  Uncorrected stacked PPL={uncorr_ppl:.3f} ({uncorr_delta:+.2f}%)")
        uncorr_ppls.append(uncorr_ppl)
        del comp_model_uncorr

        # Probe IDs (for correction fitting)
        probe_ids = owt_corpus.probe_train[:, :-1]

        # Step 3: Naive corrections
        print(f"\n  -- Naive corrections (fitted on float activations) --")
        comp_model_naive = swap_compressed_ffns(teacher, pack, inplace=False)
        naive_diags, naive_hooks = run_naive_corrections(
            teacher, comp_model_naive, probe_ids, TARGET_BLOCKS, seed
        )
        naive_metrics = _evaluate_model(teacher, comp_model_naive, owt_corpus, batch_size=BATCH_SIZE)
        naive_ppl = naive_metrics["perplexity"]
        naive_delta = 100.0 * (naive_ppl - baseline_ppl) / baseline_ppl
        print(f"  Naive-corrected PPL={naive_ppl:.3f} ({naive_delta:+.2f}%)")
        for h in naive_hooks:
            h.remove()
        naive_ppls.append(naive_ppl)
        naive_diags_all.append(naive_diags)
        del comp_model_naive

        # Step 4: Cascaded corrections
        print(f"\n  -- Cascaded corrections (each fitted on current comp_model output) --")
        comp_model_casc = swap_compressed_ffns(teacher, pack, inplace=False)
        casc_diags, casc_hooks = run_cascaded_corrections(
            teacher, comp_model_casc, probe_ids, TARGET_BLOCKS, seed
        )
        casc_metrics = _evaluate_model(teacher, comp_model_casc, owt_corpus, batch_size=BATCH_SIZE)
        casc_ppl = casc_metrics["perplexity"]
        casc_delta = 100.0 * (casc_ppl - baseline_ppl) / baseline_ppl
        print(f"  Cascaded-corrected PPL={casc_ppl:.3f} ({casc_delta:+.2f}%)")
        for h in casc_hooks:
            h.remove()
        casc_ppls.append(casc_ppl)
        casc_diags_all.append(casc_diags)
        del comp_model_casc

    # ---- Aggregate ----
    def agg(vals: list[float]) -> tuple[float, float]:
        t = torch.tensor(vals, dtype=torch.float32)
        return float(t.mean().item()), float(t.std().item()) if len(vals) > 1 else 0.0

    uncorr_mean, uncorr_std = agg(uncorr_ppls)
    naive_mean, naive_std = agg(naive_ppls)
    casc_mean, casc_std = agg(casc_ppls)

    uncorr_deltas = [100.0 * (p - baseline_ppl) / baseline_ppl for p in uncorr_ppls]
    naive_deltas = [100.0 * (p - baseline_ppl) / baseline_ppl for p in naive_ppls]
    casc_deltas = [100.0 * (p - baseline_ppl) / baseline_ppl for p in casc_ppls]

    uncorr_dm, uncorr_ds = agg(uncorr_deltas)
    naive_dm, naive_ds = agg(naive_deltas)
    casc_dm, casc_ds = agg(casc_deltas)

    # Verdict based on cascaded
    if casc_dm <= 12.0:
        verdict = "CONFIRMED"
    elif casc_dm >= uncorr_dm:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    results = {
        "cycle": 63,
        "item": 79,
        "target_blocks": TARGET_BLOCKS,
        "hidden_fraction": HIDDEN_FRACTION,
        "seeds": SEEDS,
        "probe_train_seqs": PROBE_TRAIN_SEQS,
        "ppl_float": baseline_ppl,
        "ppl_uncorrected_mean": uncorr_mean,
        "ppl_uncorrected_std": uncorr_std,
        "ppl_naive_mean": naive_mean,
        "ppl_naive_std": naive_std,
        "ppl_cascaded_mean": casc_mean,
        "ppl_cascaded_std": casc_std,
        "pct_degrad_uncorrected_mean": uncorr_dm,
        "pct_degrad_uncorrected_std": uncorr_ds,
        "pct_degrad_naive_mean": naive_dm,
        "pct_degrad_naive_std": naive_ds,
        "pct_degrad_cascaded_mean": casc_dm,
        "pct_degrad_cascaded_std": casc_ds,
        "ppl_uncorrected_per_seed": {s: v for s, v in zip(SEEDS, uncorr_ppls)},
        "ppl_naive_per_seed": {s: v for s, v in zip(SEEDS, naive_ppls)},
        "ppl_cascaded_per_seed": {s: v for s, v in zip(SEEDS, casc_ppls)},
        "pct_degrad_uncorrected_per_seed": {s: v for s, v in zip(SEEDS, uncorr_deltas)},
        "pct_degrad_naive_per_seed": {s: v for s, v in zip(SEEDS, naive_deltas)},
        "pct_degrad_cascaded_per_seed": {s: v for s, v in zip(SEEDS, casc_deltas)},
        "naive_correction_diags": naive_diags_all,
        "cascaded_correction_diags": casc_diags_all,
        "verdict": verdict,
    }
    return results, baseline_ppl


def make_plot(results: dict, out_path: Path) -> None:
    baseline_ppl = results["ppl_float"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GPT-2 FFN Stacking-Aware Correction (k=4@6x, OWT)", fontsize=13)

    # Left: PPL bar chart with error bars
    ax = axes[0]
    labels = ["Float", "Uncorrected", "Naive\n(cycle-043)", "Cascaded\n(stacking-aware)"]
    ppls = [
        baseline_ppl,
        results["ppl_uncorrected_mean"],
        results["ppl_naive_mean"],
        results["ppl_cascaded_mean"],
    ]
    stds = [0.0, results["ppl_uncorrected_std"], results["ppl_naive_std"], results["ppl_cascaded_std"]]
    colors = ["#4CAF50", "#F44336", "#FF9800", "#2196F3"]
    bars = ax.bar(labels, ppls, 0.5, yerr=stds, color=colors, alpha=0.85, capsize=5)
    ax.set_ylabel("Perplexity (lower = better)")
    ax.set_title("PPL Comparison")
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2.0, ppl + 0.5,
                f"{ppl:.2f}", ha="center", va="bottom", fontsize=9)

    # Right: per-block held-out MSE for seed=42 (first seed), naive vs cascaded
    ax2 = axes[1]
    # Use first seed's diags
    naive_diags_s0 = results["naive_correction_diags"][0] if results["naive_correction_diags"] else []
    casc_diags_s0 = results["cascaded_correction_diags"][0] if results["cascaded_correction_diags"] else []

    if naive_diags_s0 and casc_diags_s0:
        n_blocks = min(len(naive_diags_s0), len(casc_diags_s0))
        block_labels = [f"Blk {d['block_id']}" for d in naive_diags_s0[:n_blocks]]
        naive_held = [d["held_out_mse"] for d in naive_diags_s0[:n_blocks]]
        casc_held = [d["held_out_mse"] for d in casc_diags_s0[:n_blocks]]
        naive_fit = [d["fit_mse"] for d in naive_diags_s0[:n_blocks]]
        casc_fit = [d["fit_mse"] for d in casc_diags_s0[:n_blocks]]

        x = np.arange(n_blocks)
        w = 0.2
        ax2.bar(x - 1.5 * w, naive_fit, w, label="Naive fit", color="#FF9800", alpha=0.85)
        ax2.bar(x - 0.5 * w, naive_held, w, label="Naive held-out", color="#FF9800", alpha=0.5, hatch="//")
        ax2.bar(x + 0.5 * w, casc_fit, w, label="Cascaded fit", color="#2196F3", alpha=0.85)
        ax2.bar(x + 1.5 * w, casc_held, w, label="Cascaded held-out", color="#2196F3", alpha=0.5, hatch="//")
        ax2.set_xticks(x)
        ax2.set_xticklabels(block_labels)
        ax2.set_ylabel("MSE at correction point (seed=42)")
        ax2.set_title("Per-block Correction MSE (fit vs held-out)")
        ax2.legend(fontsize=8)

    uncorr_d = results["pct_degrad_uncorrected_mean"]
    naive_d = results["pct_degrad_naive_mean"]
    casc_d = results["pct_degrad_cascaded_mean"]
    verdict = results["verdict"]
    fig.text(0.5, 0.01,
             f"Uncorrected: {uncorr_d:+.1f}%  |  Naive: {naive_d:+.1f}%  |  Cascaded: {casc_d:+.1f}%  |  Verdict: {verdict}",
             ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved plot to {out_path}")


def main() -> None:
    print("Cycle 063: Stacking-Aware Boundary Correction — GPT-2 FFN")
    results, baseline_ppl = run_experiment()

    json_path = RESULTS_DIR / "cycle_063_stacking_correction.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {json_path}")

    png_path = RESULTS_DIR / "cycle_063_stacking_correction.png"
    make_plot(results, png_path)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — Cycle 063: Stacking-Aware Boundary Correction")
    print("=" * 60)
    print(f"Baseline GPT-2 PPL (OWT): {results['ppl_float']:.3f}")
    print(f"Uncorrected stacked:")
    print(f"  PPL={results['ppl_uncorrected_mean']:.3f}±{results['ppl_uncorrected_std']:.3f}  "
          f"({results['pct_degrad_uncorrected_mean']:+.2f}%±{results['pct_degrad_uncorrected_std']:.2f}%)")
    print(f"Naive corrections (cycle-043 style):")
    print(f"  PPL={results['ppl_naive_mean']:.3f}±{results['ppl_naive_std']:.3f}  "
          f"({results['pct_degrad_naive_mean']:+.2f}%±{results['pct_degrad_naive_std']:.2f}%)")
    print(f"Cascaded corrections (stacking-aware):")
    print(f"  PPL={results['ppl_cascaded_mean']:.3f}±{results['ppl_cascaded_std']:.3f}  "
          f"({results['pct_degrad_cascaded_mean']:+.2f}%±{results['pct_degrad_cascaded_std']:.2f}%)")
    print(f"\nPer-seed PPL:")
    for s in SEEDS:
        print(f"  seed={s}: uncorr={results['ppl_uncorrected_per_seed'][s]:.3f}, "
              f"naive={results['ppl_naive_per_seed'][s]:.3f}, "
              f"casc={results['ppl_cascaded_per_seed'][s]:.3f}")

    print(f"\n*** VERDICT: {results['verdict']} ***")
    print(f"  Cascaded {results['pct_degrad_cascaded_mean']:+.1f}%"
          f"  vs criterion ≤+12% (CONFIRMED) or ≥uncorrected (REFUTED)")

    # Overfitting flags
    all_casc_diags = [d for seed_diags in results["cascaded_correction_diags"] for d in seed_diags]
    flagged = [d for d in all_casc_diags if d.get("overfitting_flag")]
    if flagged:
        print(f"\nOVERFITTING FLAGS ({len(flagged)} / {len(all_casc_diags)} points):")
        for d in flagged:
            print(f"  seed=? block {d['block_id']}: fit={d['fit_mse']:.5f}, held={d['held_out_mse']:.5f}")
    else:
        print(f"\nNo overfitting flags ({len(all_casc_diags)} correction points checked).")


if __name__ == "__main__":
    main()
