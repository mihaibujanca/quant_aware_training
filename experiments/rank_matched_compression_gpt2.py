"""Cycle 073: Rank-matched adaptive compression — GPT-2 FFN (item 84).

Per-block channels proportional to rank_99 (cycle 068), total budget matched
to uniform 6x (4 blocks × 512 = 2048 channels). Compares:
  - uniform: all blocks use 512 channels
  - rank_matched: B7=481, B5=479, B4=477, B1=611 (total=2048)

Both configs run with 0 and 500 polish steps (Adam on W_down).
Polish uses held-out overfitting guard (abort if 3 consecutive checks increase).

Criterion:
  CONFIRMED if rank_matched achieves ≤+20% unpolished OR ≤+3% polished
             (vs uniform baselines +30.2% / +4.9%).
  REFUTED   if rank_matched performs worse than uniform at both conditions.
  INCONCLUSIVE otherwise.
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
TARGET_BLOCKS = [7, 5, 4, 1]   # Blocks to compress (easiest-first from cycle 058)
SEEDS = [42, 123, 7]
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200     # 200 holdout for polish + overfitting guard
EVAL_VAL_SEQS = 200
SEQUENCE_LENGTH = 128
BATCH_SIZE = 4

# Per-plan: 400 seqs total probe (200 train + 200 holdout) for gradient signal
# Increasing probe_train to 400 gives 51K tokens for B1 (611ch × 768 ≈ 469K params)
PROBE_TRAIN_SEQS_EXTENDED = 400  # used for geometry probe; first 200 = train, next 200 = holdout

POLISH_STEPS = 500
POLISH_LR = 1e-4
HOLDOUT_CHECK_EVERY = 50
HOLDOUT_ABORT_STREAK = 3

# Budget configs
UNIFORM_HIDDEN = {b: 512 for b in TARGET_BLOCKS}
RANK_MATCHED_HIDDEN = {7: 481, 5: 479, 4: 477, 1: 611}  # total=2048, from cycle 068 rank_99

# Verify budgets
assert sum(UNIFORM_HIDDEN.values()) == 2048, f"Uniform sum={sum(UNIFORM_HIDDEN.values())}"
assert sum(RANK_MATCHED_HIDDEN.values()) == 2048, f"Rank-matched sum={sum(RANK_MATCHED_HIDDEN.values())}"
print(f"Budget check: uniform={sum(UNIFORM_HIDDEN.values())}, rank_matched={sum(RANK_MATCHED_HIDDEN.values())} (both=2048 OK)")

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


def _compute_recon_error_norms(atlas: FfnGeometryAtlas, per_layer_hidden: dict[int, int]) -> dict[int, float]:
    """Per-block Frobenius reconstruction error (compressed vs original W_fc).

    We measure how much of W_in is discarded by summing the squared norms of
    the un-selected rows.
    """
    norms: dict[int, float] = {}
    for layer in atlas.layer_geometries:
        k = per_layer_hidden[layer.layer_id]
        budget = CompressionBudgetSpec(
            per_layer_hidden={layer.layer_id: k},
            selection_mode="topk",
        )
        selected = _select_hidden_units(layer, budget, k)
        all_idx = torch.arange(layer.hidden_dim)
        mask = torch.ones(layer.hidden_dim, dtype=torch.bool)
        mask[selected] = False
        dropped_idx = all_idx[mask]
        W_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
        if dropped_idx.numel() > 0:
            dropped_rows = W_in[dropped_idx]
            frob = float(dropped_rows.norm("fro").item())
        else:
            frob = 0.0
        norms[layer.layer_id] = frob
    return norms


def _find_compressed_mlp(model: nn.Module, block_id: int):
    try:
        block = model.transformer.h[block_id]
    except (AttributeError, IndexError):
        return None
    return getattr(block, "mlp", None)


def _collect_polish_params(student_model: nn.Module, block_ids: list[int]) -> list[torch.Tensor]:
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


def _polish_blocks_with_guard(
    student_model: nn.Module,
    block_ids: list[int],
    train_tokens: torch.Tensor,
    holdout_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
    check_every: int = HOLDOUT_CHECK_EVERY,
    abort_streak: int = HOLDOUT_ABORT_STREAK,
) -> tuple[list[float], list[float], int]:
    """Adam on W_down for block_ids. Returns (train_losses, holdout_losses, stopped_step).

    Overfitting guard: abort if holdout loss increases for abort_streak consecutive checks.
    """
    if n_steps == 0:
        return [], [], 0

    student_model.requires_grad_(False)
    params = _collect_polish_params(student_model, block_ids)
    if not params:
        return [], [], 0

    optimizer = torch.optim.Adam(params, lr=lr)
    student_model.train()

    train_losses: list[float] = []
    holdout_losses: list[float] = []
    streak = 0
    last_holdout = float("inf")
    stopped_step = n_steps
    n_seqs = train_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(42)

    for step in range(1, n_steps + 1):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = train_tokens[idx].to(device)
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
        train_losses.append(float(loss.item()))

        # Holdout check
        if step % check_every == 0:
            student_model.eval()
            student_model.requires_grad_(False)
            with torch.no_grad():
                ho_losses_batch = []
                for i in range(0, min(holdout_tokens.shape[0], 32), BATCH_SIZE):
                    b = holdout_tokens[i:i+BATCH_SIZE].to(device)
                    inp = b[:, :-1].contiguous()
                    tgt = b[:, 1:].contiguous()
                    raw2 = student_model(inp)
                    lg = raw2[0] if isinstance(raw2, tuple) else (raw2.logits if hasattr(raw2, "logits") else raw2)
                    l2 = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))
                    ho_losses_batch.append(float(l2.item()))
                ho_loss = float(torch.tensor(ho_losses_batch).mean().item())
            holdout_losses.append(ho_loss)
            print(f"      step {step}: train_loss={train_losses[-1]:.4f}, holdout={ho_loss:.4f}", end="")
            if ho_loss > last_holdout:
                streak += 1
                print(f" [overfitting streak={streak}]")
                if streak >= abort_streak:
                    print(f"      EARLY STOP at step {step} (overfitting guard triggered)")
                    stopped_step = step
                    # Re-enable grad=False for all
                    student_model.eval()
                    student_model.requires_grad_(False)
                    return train_losses, holdout_losses, stopped_step
            else:
                streak = 0
                print()
            last_holdout = ho_loss
            student_model.train()
            for p in params:
                p.requires_grad_(True)

    student_model.eval()
    student_model.requires_grad_(False)
    return train_losses, holdout_losses, stopped_step


def run_single_config(
    teacher: TeacherBundle,
    owt_corpus,
    per_layer_hidden: dict[int, int],
    config_name: str,
    seed: int,
) -> dict:
    """Run one (config, seed) combination. Returns metrics dict."""
    _set_seed(seed)
    device = teacher.device

    # Build budget with per-layer overrides
    budget = CompressionBudgetSpec(
        per_layer_hidden=per_layer_hidden,
        selection_mode="topk",
    )

    # Probe geometry
    print(f"\n  [{config_name}|seed={seed}] Probing FFN geometry for blocks {TARGET_BLOCKS}...")
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher,
        owt_corpus,
        layer_ids=tuple(TARGET_BLOCKS),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )

    # Compute per-block reconstruction error norms
    recon_norms = _compute_recon_error_norms(atlas, per_layer_hidden)
    print(f"  [{config_name}|seed={seed}] Recon error norms: {recon_norms}")

    # Build compressed model
    _set_seed(seed)
    pack = derive_ffn_compression(atlas, budget)
    student = swap_compressed_ffns(teacher, pack, inplace=False)
    student.eval()

    # Unpolished PPL
    metrics_0 = _evaluate_model(teacher, student, owt_corpus, batch_size=BATCH_SIZE)
    ppl_0 = metrics_0["perplexity"]
    print(f"  [{config_name}|seed={seed}] Unpolished PPL: {ppl_0:.3f}")

    # Polish
    train_tokens = owt_corpus.probe_train
    holdout_tokens = owt_corpus.probe_holdout

    train_losses, holdout_losses, stopped_step = _polish_blocks_with_guard(
        student_model=student,
        block_ids=TARGET_BLOCKS,
        train_tokens=train_tokens,
        holdout_tokens=holdout_tokens,
        n_steps=POLISH_STEPS,
        lr=POLISH_LR,
        device=device,
    )

    # Polished PPL
    metrics_500 = _evaluate_model(teacher, student, owt_corpus, batch_size=BATCH_SIZE)
    ppl_500 = metrics_500["perplexity"]
    print(f"  [{config_name}|seed={seed}] Polished PPL ({stopped_step} steps): {ppl_500:.3f}")

    return {
        "ppl_0": ppl_0,
        "ppl_500": ppl_500,
        "polish_steps_done": stopped_step,
        "train_losses": train_losses,
        "holdout_losses": holdout_losses,
        "recon_norms": {str(k): v for k, v in recon_norms.items()},
        "per_layer_hidden": {str(k): v for k, v in per_layer_hidden.items()},
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
    print(f"Teacher loaded: {teacher.model_name}, device={teacher.device}")

    # Baseline teacher PPL
    owt_corpus_base = build_probe_corpus(
        ProbeCorpusSpec(
            teacher=teacher,
            text_path=str(OWT_PATH),
            sequence_length=SEQUENCE_LENGTH,
            probe_train_sequences=PROBE_TRAIN_SEQS_EXTENDED,
            probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
            eval_val_sequences=EVAL_VAL_SEQS,
            shuffle_seed=42,
        )
    )
    baseline_metrics = _evaluate_model(teacher, teacher.model, owt_corpus_base, batch_size=BATCH_SIZE)
    baseline_ppl = baseline_metrics["perplexity"]
    print(f"Baseline GPT-2 PPL (OWT): {baseline_ppl:.3f}")

    configs = {
        "uniform": UNIFORM_HIDDEN,
        "rank_matched": RANK_MATCHED_HIDDEN,
    }

    # Results: results[config][seed] = {...}
    all_results: dict[str, dict[int, dict]] = {cfg: {} for cfg in configs}

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
                probe_train_sequences=PROBE_TRAIN_SEQS_EXTENDED,
                probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
                eval_val_sequences=EVAL_VAL_SEQS,
                shuffle_seed=seed,
            )
        )
        print(f"  probe_train shape: {owt_corpus.probe_train.shape}")
        print(f"  probe_holdout shape: {owt_corpus.probe_holdout.shape}")

        for cfg_name, per_layer_hidden in configs.items():
            metrics = run_single_config(
                teacher=teacher,
                owt_corpus=owt_corpus,
                per_layer_hidden=per_layer_hidden,
                config_name=cfg_name,
                seed=seed,
            )
            all_results[cfg_name][seed] = metrics

    # Aggregate
    def agg(vals: list[float]) -> tuple[float, float]:
        t = torch.tensor(vals, dtype=torch.float32)
        return float(t.mean().item()), float(t.std().item()) if len(t) > 1 else 0.0

    agg_results: dict[str, dict] = {}
    for cfg_name in configs:
        ppl_0s = [all_results[cfg_name][s]["ppl_0"] for s in SEEDS]
        ppl_500s = [all_results[cfg_name][s]["ppl_500"] for s in SEEDS]
        mean_0, std_0 = agg(ppl_0s)
        mean_500, std_500 = agg(ppl_500s)
        delta_0s = [100.0 * (p - baseline_ppl) / baseline_ppl for p in ppl_0s]
        delta_500s = [100.0 * (p - baseline_ppl) / baseline_ppl for p in ppl_500s]
        mean_d0, std_d0 = agg(delta_0s)
        mean_d500, std_d500 = agg(delta_500s)
        agg_results[cfg_name] = {
            "ppl_0_mean": mean_0, "ppl_0_std": std_0,
            "ppl_500_mean": mean_500, "ppl_500_std": std_500,
            "delta_pct_0_mean": mean_d0, "delta_pct_0_std": std_d0,
            "delta_pct_500_mean": mean_d500, "delta_pct_500_std": std_d500,
            "per_seed_ppl_0": {s: all_results[cfg_name][s]["ppl_0"] for s in SEEDS},
            "per_seed_ppl_500": {s: all_results[cfg_name][s]["ppl_500"] for s in SEEDS},
        }

    # Per-block recon norms (mean over seeds for each config)
    per_block_norms: dict[str, dict[str, float]] = {}
    for cfg_name in configs:
        norms_per_block: dict[str, list[float]] = {}
        for s in SEEDS:
            for bk, v in all_results[cfg_name][s]["recon_norms"].items():
                norms_per_block.setdefault(bk, []).append(v)
        per_block_norms[cfg_name] = {bk: float(torch.tensor(v).mean().item()) for bk, v in norms_per_block.items()}

    # Verdict
    rm = agg_results["rank_matched"]
    un = agg_results["uniform"]
    confirmed = (rm["delta_pct_0_mean"] <= 20.0 or rm["delta_pct_500_mean"] <= 3.0)
    refuted = (rm["delta_pct_0_mean"] > un["delta_pct_0_mean"] and
               rm["delta_pct_500_mean"] > un["delta_pct_500_mean"])
    if refuted:
        verdict = "REFUTED"
    elif confirmed:
        verdict = "CONFIRMED"
    else:
        verdict = "INCONCLUSIVE"

    results = {
        "baseline_ppl": baseline_ppl,
        "target_blocks": TARGET_BLOCKS,
        "seeds": SEEDS,
        "configs": {
            "uniform": {str(k): v for k, v in UNIFORM_HIDDEN.items()},
            "rank_matched": {str(k): v for k, v in RANK_MATCHED_HIDDEN.items()},
        },
        "agg": agg_results,
        "per_block_recon_norms": per_block_norms,
        "per_seed_details": {cfg: {s: all_results[cfg][s] for s in SEEDS} for cfg in configs},
        "verdict": verdict,
        "verdict_details": {
            "rank_matched_delta_0_mean": rm["delta_pct_0_mean"],
            "rank_matched_delta_500_mean": rm["delta_pct_500_mean"],
            "uniform_delta_0_mean": un["delta_pct_0_mean"],
            "uniform_delta_500_mean": un["delta_pct_500_mean"],
            "criterion_unpolished_threshold": 20.0,
            "criterion_polished_threshold": 3.0,
            "cycle061_reference": {"0_steps": 30.2, "500_steps": 4.9},
        },
    }
    return results


def make_plot(results: dict, out_path: Path) -> None:
    agg = results["agg"]
    baseline_ppl = results["baseline_ppl"]

    configs = ["uniform", "rank_matched"]
    labels = ["Uniform 0-step", "Rank-match 0-step", "Uniform 500-step", "Rank-match 500-step"]
    means = [
        agg["uniform"]["delta_pct_0_mean"],
        agg["rank_matched"]["delta_pct_0_mean"],
        agg["uniform"]["delta_pct_500_mean"],
        agg["rank_matched"]["delta_pct_500_mean"],
    ]
    stds = [
        agg["uniform"]["delta_pct_0_std"],
        agg["rank_matched"]["delta_pct_0_std"],
        agg["uniform"]["delta_pct_500_std"],
        agg["rank_matched"]["delta_pct_500_std"],
    ]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    verdict = results["verdict"]
    fig.suptitle(
        f"Rank-Matched Adaptive Compression — GPT-2 FFN k=4 (OWT)\n"
        f"Baseline PPL={baseline_ppl:.3f}  |  Verdict: {verdict}",
        fontsize=12,
    )

    # Left: bar chart
    ax = axes[0]
    x = np.arange(len(means))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=5, width=0.6)
    ax.axhline(20.0, color="darkorange", lw=1.5, ls="--", label="+20% unpolished target")
    ax.axhline(3.0, color="green", lw=1.5, ls="--", label="+3% polished target")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.3, f"{m:+.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, fontsize=8)
    ax.set_ylabel("PPL % change vs baseline")
    ax.set_title("PPL degradation by config")
    ax.legend(fontsize=8)

    # Right: per-block reconstruction error norms
    ax2 = axes[1]
    block_ids = [str(b) for b in results["target_blocks"]]
    norms_uniform = [results["per_block_recon_norms"]["uniform"].get(b, 0.0) for b in block_ids]
    norms_rm = [results["per_block_recon_norms"]["rank_matched"].get(b, 0.0) for b in block_ids]
    x2 = np.arange(len(block_ids))
    w = 0.35
    ax2.bar(x2 - w/2, norms_uniform, w, label="Uniform 512", color="#2196F3", alpha=0.85)
    ax2.bar(x2 + w/2, norms_rm, w, label="Rank-matched", color="#FF9800", alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"Block {b}" for b in block_ids])
    ax2.set_ylabel("Reconstruction error (Frobenius)")
    ax2.set_title("Per-block reconstruction error")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved plot to {out_path}")


def main() -> None:
    print("Cycle 073: Rank-matched adaptive compression — GPT-2 FFN (item 84)")
    print(f"Configs: uniform={UNIFORM_HIDDEN}, rank_matched={RANK_MATCHED_HIDDEN}")
    print(f"Both sum to 2048 channels across {len(TARGET_BLOCKS)} blocks")

    results = run_experiment()

    json_path = RESULTS_DIR / "cycle_073_rank_matched_compression.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {json_path}")

    png_path = RESULTS_DIR / "cycle_073_rank_matched_compression.png"
    make_plot(results, png_path)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Baseline GPT-2 PPL (OWT): {results['baseline_ppl']:.3f}")
    print(f"Cycle 061 reference: 0-steps=+30.2%, 500-steps=+4.9%")

    print(f"\n{'Config':<14} {'0-step PPL':>10} {'0-step Δ%':>10} {'500-step PPL':>12} {'500-step Δ%':>12}")
    print("-" * 62)
    for cfg in ["uniform", "rank_matched"]:
        a = results["agg"][cfg]
        print(f"{cfg:<14} {a['ppl_0_mean']:>10.3f} {a['delta_pct_0_mean']:>+10.1f}% "
              f"{a['ppl_500_mean']:>12.3f} {a['delta_pct_500_mean']:>+12.1f}%")

    vd = results["verdict_details"]
    print(f"\nVerdict: {results['verdict']}")
    print(f"  rank_matched unpolished: {vd['rank_matched_delta_0_mean']:+.1f}% (target ≤+{vd['criterion_unpolished_threshold']:.0f}%)")
    print(f"  rank_matched polished:   {vd['rank_matched_delta_500_mean']:+.1f}% (target ≤+{vd['criterion_polished_threshold']:.0f}%)")
    print(f"  uniform unpolished:      {vd['uniform_delta_0_mean']:+.1f}%")
    print(f"  uniform polished:        {vd['uniform_delta_500_mean']:+.1f}%")

    print("\nPer-block recon norms (mean over seeds):")
    for cfg in ["uniform", "rank_matched"]:
        norms = results["per_block_recon_norms"][cfg]
        print(f"  {cfg}: { {k: f'{v:.2f}' for k, v in norms.items()} }")


if __name__ == "__main__":
    main()
