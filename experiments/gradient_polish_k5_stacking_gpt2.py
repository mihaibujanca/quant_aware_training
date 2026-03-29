"""Cycle 106: Gradient polish on k=5 stacking — GPT-2 FFN.

Tests whether Adam steps on W_down (all 5 blocks jointly) can push k=5
stacking below the cascade wall observed in prior cycles.

Configs: [7,5,1,3,6] and [7,5,4,1,6] at 6x compression.
Polish budgets: {0, 50, 200, 500} Adam steps at lr=1e-4.
Seeds: [42, 123, 7].
Data: OWT (openwebtext_sample.txt).

Verdict thresholds (per plan):
  CONFIRMED     if best polish reduces OWT delta by >=50% for >=1 config.
  REFUTED       if best polish reduces delta by <20% for both configs.
  INCONCLUSIVE  if 20-50% reduction range.
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
TARGET_BLOCKS_LIST = [
    [7, 5, 1, 3, 6],
    [7, 5, 4, 1, 6],
]
HIDDEN_FRACTION = 1 / 6         # 6x compression
SEEDS = [42, 123, 7]
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200        # Used as polish data
EVAL_HOLDOUT_SEQS = 50          # Held-out split for per-step eval logging
EVAL_VAL_SEQS = 200
SEQUENCE_LENGTH = 128
BATCH_SIZE = 4

POLISH_STEPS_LIST = [0, 50, 200, 500]
POLISH_LR = 1e-4
EVAL_EVERY = 10                 # Per-step eval every N steps
OVERFITTING_THRESHOLD = 1.02    # Flag if eval PPL increases >2% from best

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


def _forward_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids)
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "logits"):
        return out.logits
    return out


def _find_compressed_mlp(model: nn.Module, block_id: int):
    try:
        block = model.transformer.h[block_id]
    except (AttributeError, IndexError):
        return None
    return getattr(block, "mlp", None)


def _eval_ppl_tokens(model: nn.Module, tokens: torch.Tensor, device: torch.device) -> float:
    """Evaluate PPL on a token tensor [n_seqs, seq_len] without updating model state."""
    model.eval()
    total_loss = 0.0
    n = tokens.shape[0]
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = tokens[i : i + BATCH_SIZE].to(device)
            inputs = batch[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()
            logits = _forward_logits(model, inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
            total_loss += float(loss.item()) * batch.shape[0]
    avg_loss = total_loss / n
    return math.exp(avg_loss)


def _polish_joint(
    student_model: nn.Module,
    block_ids: list[int],
    polish_tokens: torch.Tensor,
    eval_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
) -> dict:
    """Run n_steps of Adam jointly on all W_down params across block_ids.

    Returns dict with:
      train_losses: list[float] (one per step)
      eval_ppls: list[dict] with {step, ppl} every EVAL_EVERY steps
      overfitting_flagged: bool
    """
    if n_steps == 0:
        return {"train_losses": [], "eval_ppls": [], "overfitting_flagged": False}

    # Collect all W_down params across compressed blocks
    params_to_optimize = []
    for block_id in block_ids:
        mlp = _find_compressed_mlp(student_model, block_id)
        if mlp is None or not hasattr(mlp, "c_proj"):
            print(f"  WARNING: no c_proj found for block {block_id}, skipping")
            continue
        w_down = mlp.c_proj.weight
        w_down.requires_grad_(True)
        params_to_optimize.append(w_down)
        b_down = getattr(mlp.c_proj, "bias", None)
        if b_down is not None:
            b_down.requires_grad_(True)
            params_to_optimize.append(b_down)

    if not params_to_optimize:
        print("  WARNING: no params to optimize, skipping polish")
        return {"train_losses": [], "eval_ppls": [], "overfitting_flagged": False}

    # Freeze all; unfreeze only collected params
    student_model.requires_grad_(False)
    for p in params_to_optimize:
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    student_model.train()

    train_losses: list[float] = []
    eval_ppls: list[dict] = []
    best_eval_ppl: float = float("inf")
    overfitting_flagged = False

    n_seqs = polish_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(42)

    for step in range(1, n_steps + 1):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = polish_tokens[idx].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()

        optimizer.zero_grad()
        logits = _forward_logits(student_model, inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

        if step % EVAL_EVERY == 0 or step == n_steps:
            student_model.eval()
            with torch.no_grad():
                ep = _eval_ppl_tokens(student_model, eval_tokens, device)
            student_model.train()
            eval_ppls.append({"step": step, "ppl": ep})
            if ep < best_eval_ppl:
                best_eval_ppl = ep
            elif ep > best_eval_ppl * OVERFITTING_THRESHOLD and not overfitting_flagged:
                print(f"    [OVERFIT FLAG] step {step}: eval PPL {ep:.3f} > best {best_eval_ppl:.3f} * {OVERFITTING_THRESHOLD}")
                overfitting_flagged = True
            print(f"    step {step}/{n_steps}: train_loss={float(loss.item()):.4f}, eval_ppl={ep:.3f}")

    student_model.eval()
    student_model.requires_grad_(False)

    return {
        "train_losses": train_losses,
        "eval_ppls": eval_ppls,
        "overfitting_flagged": overfitting_flagged,
    }


def run_one_config(
    target_blocks: list[int],
    teacher: TeacherBundle,
    baseline_ppl: float,
    device: torch.device,
) -> dict:
    """Run the full sweep for one TARGET_BLOCKS config. Returns per-seed/per-steps results."""

    config_label = str(target_blocks)
    print(f"\n{'#'*70}")
    print(f"CONFIG: {config_label}")
    print(f"{'#'*70}")

    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        min_hidden=32,
        selection_mode="topk",
    )

    # per_seed_results[seed][n_steps] = {ppl, delta_pct, polish_log}
    per_seed_results: dict[int, dict[int, dict]] = {s: {} for s in SEEDS}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed} | CONFIG {config_label}")
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

        # Carve off 50 seqs from probe_holdout for per-step eval
        polish_tokens = owt_corpus.probe_holdout[:PROBE_HOLDOUT_SEQS - EVAL_HOLDOUT_SEQS]
        eval_tokens = owt_corpus.probe_holdout[PROBE_HOLDOUT_SEQS - EVAL_HOLDOUT_SEQS:]
        print(f"  polish_tokens: {polish_tokens.shape}, eval_tokens: {eval_tokens.shape}")

        print(f"  Probing FFN geometry for blocks {target_blocks}...")
        _set_seed(seed)
        atlas = probe_ffn_geometry(
            teacher,
            owt_corpus,
            layer_ids=tuple(target_blocks),
            max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
            batch_size=BATCH_SIZE,
        )

        for n_steps in POLISH_STEPS_LIST:
            print(f"\n  -- steps={n_steps} --")
            _set_seed(seed)

            # Build fresh student from scratch (pure lstsq)
            pack_all = derive_ffn_compression(atlas, budget)
            student = swap_compressed_ffns(teacher, pack_all, inplace=False)
            student.eval()

            if n_steps > 0:
                print(f"  Joint-polishing {len(target_blocks)} W_down for {n_steps} steps...")
                polish_log = _polish_joint(
                    student,
                    target_blocks,
                    polish_tokens,
                    eval_tokens,
                    n_steps=n_steps,
                    lr=POLISH_LR,
                    device=device,
                )
                if polish_log["train_losses"]:
                    print(f"    train loss: {polish_log['train_losses'][0]:.4f} → {polish_log['train_losses'][-1]:.4f}")
                    # Validate: train loss decreasing for first 10 steps
                    if len(polish_log["train_losses"]) >= 10:
                        first10 = polish_log["train_losses"][:10]
                        if first10[-1] > first10[0]:
                            print(f"    WARNING: train loss NOT decreasing in first 10 steps!")
            else:
                polish_log = {"train_losses": [], "eval_ppls": [], "overfitting_flagged": False}

            metrics = _evaluate_model(teacher, student, owt_corpus, batch_size=BATCH_SIZE)
            ppl = metrics["perplexity"]
            delta_pct = 100.0 * (ppl - baseline_ppl) / baseline_ppl
            print(f"  steps={n_steps}: PPL={ppl:.3f} ({delta_pct:+.1f}%)")

            per_seed_results[seed][n_steps] = {
                "ppl": ppl,
                "delta_pct": delta_pct,
                "polish_log": polish_log,
            }

    # Aggregate over seeds
    def agg(vals: list[float]) -> tuple[float, float]:
        t = torch.tensor(vals, dtype=torch.float32)
        return float(t.mean().item()), (float(t.std().item()) if len(vals) > 1 else 0.0)

    agg_results: dict[int, dict] = {}
    for n_steps in POLISH_STEPS_LIST:
        ppls = [per_seed_results[s][n_steps]["ppl"] for s in SEEDS]
        deltas = [per_seed_results[s][n_steps]["delta_pct"] for s in SEEDS]
        mean_ppl, std_ppl = agg(ppls)
        mean_delta, std_delta = agg(deltas)
        agg_results[n_steps] = {
            "ppl_mean": mean_ppl,
            "ppl_std": std_ppl,
            "delta_pct_mean": mean_delta,
            "delta_pct_std": std_delta,
            "per_seed": {s: per_seed_results[s][n_steps]["ppl"] for s in SEEDS},
            "per_seed_delta": {s: per_seed_results[s][n_steps]["delta_pct"] for s in SEEDS},
        }

    # Relative improvement vs 0-step baseline
    baseline_delta = agg_results[0]["delta_pct_mean"]
    for n_steps in POLISH_STEPS_LIST:
        if abs(baseline_delta) > 1e-6:
            rel = 100.0 * (baseline_delta - agg_results[n_steps]["delta_pct_mean"]) / abs(baseline_delta)
        else:
            rel = float("nan")
        agg_results[n_steps]["relative_improvement_pct"] = rel

    best_rel = max(
        (agg_results[n]["relative_improvement_pct"] for n in POLISH_STEPS_LIST if n > 0),
        default=float("nan"),
    )

    return {
        "config": target_blocks,
        "baseline_ppl": baseline_ppl,
        "baseline_delta_pct": agg_results[0]["delta_pct_mean"],
        "agg": agg_results,
        "per_seed": {s: per_seed_results[s] for s in SEEDS},
        "best_relative_improvement_pct": best_rel,
    }


def run_experiment() -> dict:
    print("Loading GPT-2 teacher...")
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(
            source="hf_pretrained",
            hf_model_name="gpt2",
            device="cpu",
        )
    )
    device = teacher.device
    print(f"Teacher loaded: {teacher.model_name}, device={device}")

    # Baseline OWT PPL
    print("Building OWT corpus for baseline PPL...")
    baseline_corpus = build_probe_corpus(
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
    baseline_metrics = _evaluate_model(teacher, teacher.model, baseline_corpus, batch_size=BATCH_SIZE)
    baseline_ppl = baseline_metrics["perplexity"]
    print(f"Baseline GPT-2 PPL (OWT): {baseline_ppl:.3f}")

    if not (25 <= baseline_ppl <= 65):
        raise RuntimeError(f"Baseline PPL {baseline_ppl:.3f} outside expected range [25, 65]. Abort.")

    # Run both configs
    config_results = []
    for target_blocks in TARGET_BLOCKS_LIST:
        result = run_one_config(target_blocks, teacher, baseline_ppl, device)
        config_results.append(result)
        # Phase 1 sanity check on 0-step delta
        zero_step_delta = result["agg"][0]["delta_pct_mean"]
        print(f"\nPhase 1 check: config {target_blocks} 0-step delta = {zero_step_delta:+.1f}%")
        if zero_step_delta < 10.0:
            print(f"  WARNING: 0-step delta <10% for config {target_blocks}. Check compression setup.")

    # Global verdict
    best_rels = {str(r["config"]): r["best_relative_improvement_pct"] for r in config_results}
    print(f"\nBest relative improvements: {best_rels}")

    confirmed = any(r["best_relative_improvement_pct"] >= 50.0 for r in config_results)
    refuted = all(r["best_relative_improvement_pct"] < 20.0 for r in config_results)

    if confirmed:
        verdict = "CONFIRMED"
    elif refuted:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "baseline_ppl": baseline_ppl,
        "target_blocks_list": TARGET_BLOCKS_LIST,
        "hidden_fraction": HIDDEN_FRACTION,
        "seeds": SEEDS,
        "polish_steps": POLISH_STEPS_LIST,
        "polish_lr": POLISH_LR,
        "config_results": config_results,
        "best_relative_improvements": best_rels,
        "verdict": verdict,
    }


def make_plot(results: dict, out_path: Path) -> None:
    configs = results["target_blocks_list"]
    steps = results["polish_steps"]
    baseline_ppl = results["baseline_ppl"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Gradient Polish k=5 Stacking — GPT-2 FFN 6× (OWT)\n"
        f"Baseline PPL={baseline_ppl:.3f} | Verdict: {results['verdict']}",
        fontsize=12,
    )
    colors = ["#2196F3", "#E53935"]

    for ax_idx, (ax, cr) in enumerate(zip(axes, results["config_results"])):
        agg = cr["agg"]
        delta_means = [agg[n]["delta_pct_mean"] for n in steps]
        delta_stds = [agg[n]["delta_pct_std"] for n in steps]
        rel_imprs = [agg[n]["relative_improvement_pct"] for n in steps]

        ax.errorbar(
            steps, delta_means, yerr=delta_stds,
            fmt="o-", color=colors[ax_idx], linewidth=2, markersize=7, capsize=5,
            label=f"config {cr['config']}",
        )
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.axhline(delta_means[0] * 0.5, color="green", lw=1.5, ls="--",
                   label="50% reduction target")
        ax.set_xlabel("Polish steps (Adam)")
        ax.set_ylabel("PPL % change vs baseline")
        ax.set_title(f"Config {cr['config']}\n0-step delta={delta_means[0]:+.1f}%, best rel={cr['best_relative_improvement_pct']:.1f}%")
        ax.legend(fontsize=9)

        for s, m, ri in zip(steps, delta_means, rel_imprs):
            if not math.isnan(ri):
                ax.text(s, m + 0.5, f"ri={ri:.0f}%", ha="center", fontsize=8, color=colors[ax_idx])

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved plot to {out_path}")


def main() -> None:
    print("Cycle 106: Gradient polish on k=5 stacking — GPT-2 FFN")
    results = run_experiment()

    json_path = RESULTS_DIR / "cycle_106_k5_polish.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved results to {json_path}")

    png_path = RESULTS_DIR / "cycle_106_k5_polish.png"
    make_plot(results, png_path)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — Cycle 106")
    print("=" * 70)
    print(f"Baseline GPT-2 PPL (OWT): {results['baseline_ppl']:.3f}")
    print(f"Polish LR: {results['polish_lr']}")

    for cr in results["config_results"]:
        print(f"\nConfig {cr['config']}:")
        print(f"  0-step delta: {cr['baseline_delta_pct']:+.1f}%")
        print(f"  {'Steps':<8} {'PPL mean':<12} {'Delta%':<12} {'RelImpr%':<12}")
        print(f"  {'-'*44}")
        for n in results["polish_steps"]:
            d = cr["agg"][n]
            ri = d["relative_improvement_pct"]
            ri_str = f"{ri:.1f}%" if not math.isnan(ri) else "baseline"
            print(f"  {n:<8} {d['ppl_mean']:<12.3f} {d['delta_pct_mean']:+.2f}%{' ':<5} {ri_str}")

    print(f"\nBest relative improvements: {results['best_relative_improvements']}")
    print(f"\n*** VERDICT: {results['verdict']} ***")
    if results["verdict"] == "CONFIRMED":
        print("Polish reduces k=5 stacking PPL delta by >=50% for >=1 config.")
    elif results["verdict"] == "REFUTED":
        print("Polish <20% reduction for both configs. W_down-only polish insufficient.")
        print("Next step: attention-inclusive polish (W_down + W_V or full-block).")
    else:
        print("20-50% range. Partial improvement, not conclusive.")


if __name__ == "__main__":
    main()
