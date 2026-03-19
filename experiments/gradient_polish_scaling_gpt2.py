"""Cycle 061: Gradient polish scaling — GPT-2 FFN (item 77).

Extends cycle 059's sweep to {100, 200, 500} steps and adds joint polish
(all 4 W_down updated simultaneously in one optimizer).

Criterion:
  CONFIRMED if 500 steps (either variant) achieves <= +8% PPL degradation
             OR joint outperforms independent by >= 5pp at matched step count.
  REFUTED   if 200 steps shows < 2pp improvement over 50 steps (plateau).
  INCONCLUSIVE otherwise.

Cycle 059 baselines: 0 steps=+30.2%, 50 steps=+18.4%.
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
HIDDEN_FRACTION = 1 / 6         # 6x compression
SEEDS = [42, 123, 7]
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200        # Polish data (4096 tokens ~ 32 seqs x 128)
EVAL_VAL_SEQS = 200
SEQUENCE_LENGTH = 128
BATCH_SIZE = 4

# 0 and 50 are replication baselines from cycle 059
POLISH_STEPS_LIST = [0, 50, 100, 200, 500]
# Intermediate eval checkpoints (for logging convergence)
EVAL_CHECKPOINTS = {50, 100, 200, 500}
POLISH_LR = 1e-4

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


def _find_compressed_mlp(model: nn.Module, block_id: int):
    """Return the CompressedMLP module for a given block_id, or None."""
    try:
        block = model.transformer.h[block_id]
    except (AttributeError, IndexError):
        return None
    return getattr(block, "mlp", None)


def _collect_polish_params(student_model: nn.Module, block_ids: list[int]) -> list[torch.Tensor]:
    """Collect c_proj.weight (and bias if present) for given block_ids."""
    params = []
    for block_id in block_ids:
        mlp = _find_compressed_mlp(student_model, block_id)
        if mlp is None or not hasattr(mlp, "c_proj"):
            print(f"  WARNING: no c_proj found for block {block_id}, skipping")
            continue
        w = mlp.c_proj.weight
        w.requires_grad_(True)
        params.append(w)
        b = getattr(mlp.c_proj, "bias", None)
        if b is not None:
            b.requires_grad_(True)
            params.append(b)
    return params


def _polish_blocks(
    student_model: nn.Module,
    block_ids: list[int],
    polish_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """Run n_steps of Adam on W_down (c_proj.weight) for one or more blocks.

    If len(block_ids) > 1, this is joint polish (single optimizer over all).
    Returns per-step losses.
    """
    if n_steps == 0:
        return []

    # Freeze all
    student_model.requires_grad_(False)
    params = _collect_polish_params(student_model, block_ids)
    if not params:
        return []

    optimizer = torch.optim.Adam(params, lr=lr)
    student_model.train()
    losses = []
    n_seqs = polish_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(42)

    for step in range(n_steps):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = polish_tokens[idx].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()

        optimizer.zero_grad()
        raw = student_model(inputs)
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    student_model.eval()
    student_model.requires_grad_(False)
    return losses


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

    # Baseline teacher PPL
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

    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        min_hidden=32,
        selection_mode="topk",
    )

    # Results: polish_ppls[seed][variant][n_steps] = ppl
    # variants: "independent", "joint"
    VARIANTS = ["independent", "joint"]
    polish_ppls: dict[int, dict[str, dict[int, float]]] = {
        s: {v: {} for v in VARIANTS} for s in SEEDS
    }
    polish_losses: dict[int, dict[str, dict[int, list[float]]]] = {
        s: {v: {} for v in VARIANTS} for s in SEEDS
    }

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
        print(f"  probe_holdout shape: {owt_corpus.probe_holdout.shape} (polish data)")

        # Probe geometry for all 4 target blocks
        print(f"\n  -- Probing FFN geometry for blocks {TARGET_BLOCKS} --")
        _set_seed(seed)
        atlas = probe_ffn_geometry(
            teacher,
            owt_corpus,
            layer_ids=tuple(TARGET_BLOCKS),
            max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
            batch_size=BATCH_SIZE,
        )

        polish_tokens = owt_corpus.probe_holdout
        print(f"  Polish tokens: {polish_tokens.shape} = {polish_tokens.numel()} tokens")

        for n_steps in POLISH_STEPS_LIST:
            print(f"\n  -- Polish steps = {n_steps} --")

            # ---- Independent variant ----
            print(f"    [independent] polishing each block sequentially...")
            _set_seed(seed)
            pack_all = derive_ffn_compression(atlas, budget)
            student_indep = swap_compressed_ffns(teacher, pack_all, inplace=False)
            student_indep.eval()

            if n_steps > 0:
                block_losses = []
                for block_id in TARGET_BLOCKS:
                    losses = _polish_blocks(
                        student_indep,
                        [block_id],
                        polish_tokens,
                        n_steps=n_steps,
                        lr=POLISH_LR,
                        device=teacher.device,
                    )
                    first_l = losses[0] if losses else float("nan")
                    last_l = losses[-1] if losses else float("nan")
                    print(f"      block {block_id}: {first_l:.4f} → {last_l:.4f}")
                    block_losses.append(losses)
                polish_losses[seed]["independent"][n_steps] = block_losses
            else:
                polish_losses[seed]["independent"][n_steps] = []

            metrics_indep = _evaluate_model(teacher, student_indep, owt_corpus, batch_size=BATCH_SIZE)
            ppl_indep = metrics_indep["perplexity"]
            polish_ppls[seed]["independent"][n_steps] = ppl_indep
            delta_pct_indep = 100.0 * (ppl_indep - baseline_ppl) / baseline_ppl
            print(f"    [independent] steps={n_steps}: PPL={ppl_indep:.3f} ({delta_pct_indep:+.1f}%)")

            # ---- Joint variant (skip n_steps=0, same as independent) ----
            if n_steps == 0:
                # 0-step: identical to independent (no polish)
                polish_ppls[seed]["joint"][0] = ppl_indep
                polish_losses[seed]["joint"][0] = []
            else:
                print(f"    [joint] polishing all 4 blocks simultaneously...")
                _set_seed(seed)
                pack_all2 = derive_ffn_compression(atlas, budget)
                student_joint = swap_compressed_ffns(teacher, pack_all2, inplace=False)
                student_joint.eval()

                losses_joint = _polish_blocks(
                    student_joint,
                    TARGET_BLOCKS,
                    polish_tokens,
                    n_steps=n_steps,
                    lr=POLISH_LR,
                    device=teacher.device,
                )
                first_l = losses_joint[0] if losses_joint else float("nan")
                last_l = losses_joint[-1] if losses_joint else float("nan")
                print(f"      joint: {first_l:.4f} → {last_l:.4f}")
                polish_losses[seed]["joint"][n_steps] = losses_joint

                metrics_joint = _evaluate_model(teacher, student_joint, owt_corpus, batch_size=BATCH_SIZE)
                ppl_joint = metrics_joint["perplexity"]
                polish_ppls[seed]["joint"][n_steps] = ppl_joint
                delta_pct_joint = 100.0 * (ppl_joint - baseline_ppl) / baseline_ppl
                print(f"    [joint] steps={n_steps}: PPL={ppl_joint:.3f} ({delta_pct_joint:+.1f}%)")

    # ---- Aggregate over seeds ----
    def agg(vals: list[float]) -> tuple[float, float]:
        t = torch.tensor(vals, dtype=torch.float32)
        return float(t.mean().item()), float(t.std().item()) if len(vals) > 1 else 0.0

    polish_agg: dict[str, dict[int, dict]] = {v: {} for v in VARIANTS}
    for variant in VARIANTS:
        for n_steps in POLISH_STEPS_LIST:
            ppls = [polish_ppls[s][variant][n_steps] for s in SEEDS]
            mean_ppl, std_ppl = agg(ppls)
            deltas = [100.0 * (p - baseline_ppl) / baseline_ppl for p in ppls]
            mean_delta, std_delta = agg(deltas)
            polish_agg[variant][n_steps] = {
                "ppl_mean": mean_ppl,
                "ppl_std": std_ppl,
                "delta_pct_mean": mean_delta,
                "delta_pct_std": std_delta,
                "per_seed": {s: polish_ppls[s][variant][n_steps] for s in SEEDS},
            }

    # Relative improvement vs 0-step baseline (shared, since 0-step is identical for both)
    baseline_delta = polish_agg["independent"][0]["delta_pct_mean"]
    for variant in VARIANTS:
        for n_steps in POLISH_STEPS_LIST:
            if abs(baseline_delta) > 1e-6:
                rel = 100.0 * (baseline_delta - polish_agg[variant][n_steps]["delta_pct_mean"]) / abs(baseline_delta)
            else:
                rel = float("nan")
            polish_agg[variant][n_steps]["relative_improvement_pct"] = rel

    # ---- Verdict ----
    delta_500_indep = polish_agg["independent"][500]["delta_pct_mean"]
    delta_500_joint = polish_agg["joint"][500]["delta_pct_mean"]
    delta_200_indep = polish_agg["independent"][200]["delta_pct_mean"]
    delta_50_indep = polish_agg["independent"][50]["delta_pct_mean"]

    # Joint vs independent advantage at matched steps (use 200 as the main comparison)
    joint_advantage_200 = delta_200_indep - polish_agg["joint"][200]["delta_pct_mean"]
    joint_advantage_500 = delta_500_indep - delta_500_joint

    confirmed = (
        delta_500_indep <= 8.0
        or delta_500_joint <= 8.0
        or joint_advantage_200 >= 5.0
        or joint_advantage_500 >= 5.0
    )
    plateau = (delta_50_indep - delta_200_indep) < 2.0

    if confirmed:
        verdict = "CONFIRMED"
    elif plateau:
        verdict = "REFUTED_PLATEAU"
    else:
        verdict = "INCONCLUSIVE"

    results = {
        "baseline_ppl": baseline_ppl,
        "target_blocks": TARGET_BLOCKS,
        "hidden_fraction": HIDDEN_FRACTION,
        "seeds": SEEDS,
        "polish_steps": POLISH_STEPS_LIST,
        "polish_lr": POLISH_LR,
        "polish_agg": polish_agg,
        "verdict": verdict,
        "verdict_details": {
            "delta_500_independent": delta_500_indep,
            "delta_500_joint": delta_500_joint,
            "joint_advantage_200": joint_advantage_200,
            "joint_advantage_500": joint_advantage_500,
            "plateau_detected": plateau,
        },
        # Cycle 059 reference baselines
        "cycle059_reference": {
            "0_steps": 30.2,
            "50_steps": 18.4,
        },
    }
    return results


def make_plot(results: dict, out_path: Path) -> None:
    steps = results["polish_steps"]
    baseline_ppl = results["baseline_ppl"]
    agg = results["polish_agg"]

    indep_deltas = [agg["independent"][n]["delta_pct_mean"] for n in steps]
    indep_stds = [agg["independent"][n]["delta_pct_std"] for n in steps]
    joint_deltas = [agg["joint"][n]["delta_pct_mean"] for n in steps]
    joint_stds = [agg["joint"][n]["delta_pct_std"] for n in steps]

    indep_rel = [agg["independent"][n]["relative_improvement_pct"] for n in steps if n > 0]
    joint_rel = [agg["joint"][n]["relative_improvement_pct"] for n in steps if n > 0]
    nonzero_steps = [n for n in steps if n > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Gradient Polish Scaling — GPT-2 FFN k=4@6× (OWT)\n"
        f"Baseline PPL={baseline_ppl:.3f}  |  Verdict: {results['verdict']}",
        fontsize=12,
    )

    # Left: PPL delta % vs steps for both variants
    ax = axes[0]
    x = np.arange(len(steps))
    w = 0.35
    ax.bar(x - w/2, indep_deltas, w, yerr=indep_stds, label="independent", color="#2196F3", alpha=0.85, capsize=4)
    ax.bar(x + w/2, joint_deltas, w, yerr=joint_stds, label="joint", color="#FF9800", alpha=0.85, capsize=4)
    ax.axhline(8.0, color="green", lw=1.5, ls="--", label="+8% target")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}" for n in steps])
    ax.set_xlabel("Polish steps (Adam)")
    ax.set_ylabel("PPL % change vs baseline")
    ax.set_title("PPL degradation after polish")
    ax.legend(fontsize=9)
    # Labels
    for i, (m_i, m_j) in enumerate(zip(indep_deltas, joint_deltas)):
        ax.text(i - w/2, m_i + 0.5, f"{m_i:+.1f}%", ha="center", fontsize=7)
        ax.text(i + w/2, m_j + 0.5, f"{m_j:+.1f}%", ha="center", fontsize=7)

    # Right: relative improvement % vs steps, two lines
    ax2 = axes[1]
    ax2.plot(nonzero_steps, indep_rel, "o-", color="#2196F3", linewidth=2, markersize=7, label="independent")
    ax2.plot(nonzero_steps, joint_rel, "s-", color="#FF9800", linewidth=2, markersize=7, label="joint")
    ax2.axhline(0, color="gray", lw=0.6, ls="--")
    # Cycle 061 thresholds
    ax2.axhline(
        100.0 * (30.2 - 8.0) / 30.2,  # where 500 steps reach +8% from +30.2 baseline
        color="green", lw=1.5, ls="--", label=f"+8% threshold ({100*(30.2-8.0)/30.2:.0f}% rel.)"
    )
    ax2.set_xlabel("Polish steps (Adam)")
    ax2.set_ylabel("Relative improvement vs 0-step baseline (%)")
    ax2.set_title("Relative PPL degradation reduction")
    ax2.legend(fontsize=9)
    for s, ri, rj in zip(nonzero_steps, indep_rel, joint_rel):
        ax2.text(s, ri + 0.8, f"{ri:.0f}%", ha="center", fontsize=7, color="#2196F3")
        ax2.text(s, rj - 2.5, f"{rj:.0f}%", ha="center", fontsize=7, color="#FF9800")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved plot to {out_path}")


def main() -> None:
    print("Cycle 061: Gradient polish scaling — GPT-2 FFN (item 77)")
    print(f"Steps: {POLISH_STEPS_LIST}, Variants: independent + joint")
    results = run_experiment()

    json_path = RESULTS_DIR / "cycle_061_gradient_polish_scaling.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {json_path}")

    png_path = RESULTS_DIR / "cycle_061_gradient_polish_scaling.png"
    make_plot(results, png_path)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Baseline GPT-2 PPL (OWT): {results['baseline_ppl']:.3f}")
    print(f"Target blocks (k=4): {results['target_blocks']}")
    print(f"Cycle 059 reference: 0-steps=+{results['cycle059_reference']['0_steps']}%, "
          f"50-steps=+{results['cycle059_reference']['50_steps']}%")

    print(f"\n{'Steps':<8} {'Indep PPL':>10} {'Indep Δ%':>10} {'Joint PPL':>10} {'Joint Δ%':>10} {'Adv':>8}")
    print("-" * 60)
    for n in results["polish_steps"]:
        di = results["polish_agg"]["independent"][n]
        dj = results["polish_agg"]["joint"][n]
        adv = di["delta_pct_mean"] - dj["delta_pct_mean"]
        print(f"{n:<8} {di['ppl_mean']:>10.3f} {di['delta_pct_mean']:>+10.1f}% "
              f"{dj['ppl_mean']:>10.3f} {dj['delta_pct_mean']:>+10.1f}% {adv:>+7.1f}pp")

    vd = results["verdict_details"]
    print(f"\nVerdict details:")
    print(f"  500-step independent: {vd['delta_500_independent']:+.1f}% (target ≤+8%)")
    print(f"  500-step joint:       {vd['delta_500_joint']:+.1f}% (target ≤+8%)")
    print(f"  Joint advantage @200: {vd['joint_advantage_200']:+.1f}pp (target ≥+5pp)")
    print(f"  Joint advantage @500: {vd['joint_advantage_500']:+.1f}pp (target ≥+5pp)")
    print(f"  Plateau detected:     {vd['plateau_detected']}")
    print(f"\n*** VERDICT: {results['verdict']} ***")


if __name__ == "__main__":
    main()
