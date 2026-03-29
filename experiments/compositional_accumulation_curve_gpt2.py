"""Cycle 089: Compositional error accumulation curve (k=1..8 blocks) — GPT-2 FFN.

For k=1..8 compressed blocks (6x, topk+lstsq + 500-step Adam polish),
measures how PPL degrades as more blocks are stacked simultaneously.
Tests two orderings: forward (least-damage-first) and reverse.
Fits additive (linear), multiplicative (exponential), and power-law scaling laws.

Hypothesis: sub-multiplicative if PPL(k) < baseline * (1+delta)^k.
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
FORWARD_ORDER = [7, 5, 4, 1, 3, 6, 8, 9]   # Least-damage-first (from cycle 005/007)
REVERSE_ORDER = list(reversed(FORWARD_ORDER))

HIDDEN_FRACTION = 1 / 6         # 6x compression
SEEDS = [42, 123, 7]
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200        # Polish data
EVAL_VAL_SEQS = 200
SEQUENCE_LENGTH = 128
BATCH_SIZE = 4

POLISH_STEPS = 500
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


def _param_count(w: torch.Tensor, b: torch.Tensor | None) -> int:
    return w.numel() + (0 if b is None else b.numel())


def _derive_block_pack_from_atlas(
    atlas: FfnGeometryAtlas,
    budget: CompressionBudgetSpec,
    block_ids: list[int],
) -> CompressedFfnPack:
    """Build a CompressedFfnPack for the specified block_ids from atlas."""
    compressed_layers = []
    for layer in atlas.layer_geometries:
        if layer.layer_id not in block_ids:
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


def _find_compressed_mlp(model: nn.Module, block_id: int):
    try:
        block = model.transformer.h[block_id]
    except (AttributeError, IndexError):
        return None
    return getattr(block, "mlp", None)


def _polish_block_v2(
    student_model: nn.Module,
    block_id: int,
    polish_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """Run n_steps of Adam on W_down (c_proj.weight) of the given compressed block."""
    if n_steps == 0:
        return []

    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        print(f"  WARNING: no c_proj found for block {block_id}, skipping polish")
        return []

    student_model.requires_grad_(False)
    w_down = mlp.c_proj.weight
    w_down.requires_grad_(True)
    b_down = getattr(mlp.c_proj, "bias", None)
    if b_down is not None:
        b_down.requires_grad_(True)

    params_to_optimize = [w_down] + ([b_down] if b_down is not None else [])
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

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


def _extract_polished_weights(student_model: nn.Module, block_id: int) -> dict:
    """Extract polished W_down (and bias) from a swapped student model."""
    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        return {}
    return {
        "c_proj_weight": mlp.c_proj.weight.detach().clone(),
        "c_proj_bias": (mlp.c_proj.bias.detach().clone() if hasattr(mlp.c_proj, "bias") and mlp.c_proj.bias is not None else None),
    }


def _inject_polished_weights(student_model: nn.Module, block_id: int, polished: dict) -> None:
    """Inject polished W_down weights back into a student model."""
    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        return
    with torch.no_grad():
        mlp.c_proj.weight.copy_(polished["c_proj_weight"])
        if polished["c_proj_bias"] is not None and hasattr(mlp.c_proj, "bias") and mlp.c_proj.bias is not None:
            mlp.c_proj.bias.copy_(polished["c_proj_bias"])


# ---------------------------------------------------------------------------
# Scaling law fitting
# ---------------------------------------------------------------------------

def _fit_additive(ks: list[int], ppls: list[float]) -> tuple[float, float, float, float]:
    """Fit PPL(k) = a + b*k via least squares. Returns (a, b, r2, rmse)."""
    ks_t = torch.tensor(ks, dtype=torch.float64)
    ppls_t = torch.tensor(ppls, dtype=torch.float64)
    # [k, 1] design matrix: [1, k]
    X = torch.stack([torch.ones_like(ks_t), ks_t], dim=1)  # [n, 2]
    # lstsq
    coeffs, _, _, _ = torch.linalg.lstsq(X, ppls_t.unsqueeze(1), rcond=None)
    a, b = float(coeffs[0, 0].item()), float(coeffs[1, 0].item())
    pred = a + b * ks_t
    ss_res = float(((ppls_t - pred) ** 2).sum().item())
    ss_tot = float(((ppls_t - ppls_t.mean()) ** 2).sum().item())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    rmse = math.sqrt(ss_res / len(ks))
    return a, b, r2, rmse


def _fit_multiplicative(ks: list[int], ppls: list[float]) -> tuple[float, float, float, float]:
    """Fit PPL(k) = a * (1+c)^k via log-linearization. Returns (a, c, r2, rmse)."""
    ks_t = torch.tensor(ks, dtype=torch.float64)
    ppls_t = torch.tensor(ppls, dtype=torch.float64)
    log_ppls = torch.log(ppls_t)
    # log(PPL) = log(a) + k * log(1+c)
    X = torch.stack([torch.ones_like(ks_t), ks_t], dim=1)
    coeffs, _, _, _ = torch.linalg.lstsq(X, log_ppls.unsqueeze(1), rcond=None)
    log_a, log_1pc = float(coeffs[0, 0].item()), float(coeffs[1, 0].item())
    a = math.exp(log_a)
    c = math.exp(log_1pc) - 1.0
    pred_log = log_a + log_1pc * ks_t
    pred = torch.exp(pred_log)
    ss_res = float(((ppls_t - pred) ** 2).sum().item())
    ss_tot = float(((ppls_t - ppls_t.mean()) ** 2).sum().item())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    rmse = math.sqrt(ss_res / len(ks))
    return a, c, r2, rmse


def _fit_power_law(ks: list[int], ppls: list[float]) -> tuple[float, float, float, float]:
    """Fit PPL(k) = a * k^b + offset (via log on k, with shift). Returns (a, b, r2, rmse).
    Simplified: fit log(PPL - offset) = log(a) + b*log(k), offset = PPL(k=0) = baseline.
    We absorb baseline into the fit: fit log(PPL) = log(a) + b*log(k+1).
    """
    ks_t = torch.tensor([math.log(k + 1) for k in ks], dtype=torch.float64)
    ppls_t = torch.tensor(ppls, dtype=torch.float64)
    log_ppls = torch.log(ppls_t)
    X = torch.stack([torch.ones_like(ks_t), ks_t], dim=1)
    coeffs, _, _, _ = torch.linalg.lstsq(X, log_ppls.unsqueeze(1), rcond=None)
    log_a, b = float(coeffs[0, 0].item()), float(coeffs[1, 0].item())
    a = math.exp(log_a)
    pred_log = log_a + b * ks_t
    pred = torch.exp(pred_log)
    ss_res = float(((ppls_t - pred) ** 2).sum().item())
    ss_tot = float(((ppls_t - ppls_t.mean()) ** 2).sum().item())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    rmse = math.sqrt(ss_res / len(ks))
    return a, b, r2, rmse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment() -> dict:
    print("Loading GPT-2 teacher...")
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2", device="cpu")
    )
    print(f"Teacher loaded: {teacher.model_name}, device={teacher.device}")

    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        min_hidden=32,
        selection_mode="topk",
    )

    ALL_BLOCKS = list(set(FORWARD_ORDER))

    # Results: [seed][ordering][k] -> ppl
    all_results: dict[int, dict[str, dict[int, float]]] = {
        s: {"forward": {}, "reverse": {}} for s in SEEDS
    }
    baseline_ppls: dict[int, float] = {}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"SEED {seed} ({seed_idx+1}/{len(SEEDS)})")
        print(f"{'='*60}")
        _set_seed(seed)

        # Build OWT corpus
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
        print(f"  Corpus: probe={owt_corpus.probe_train.shape}, holdout={owt_corpus.probe_holdout.shape}, eval={owt_corpus.eval_val.shape}")

        # Baseline PPL
        bl_metrics = _evaluate_model(teacher, teacher.model, owt_corpus, batch_size=BATCH_SIZE)
        baseline_ppl = bl_metrics["perplexity"]
        baseline_ppls[seed] = baseline_ppl
        print(f"  Baseline PPL (OWT): {baseline_ppl:.3f}")

        # Probe geometry for all 8 blocks at once
        print(f"\n  Probing FFN geometry for all {len(ALL_BLOCKS)} blocks: {sorted(ALL_BLOCKS)}")
        _set_seed(seed)
        atlas = probe_ffn_geometry(
            teacher,
            owt_corpus,
            layer_ids=tuple(sorted(ALL_BLOCKS)),
            max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
            batch_size=BATCH_SIZE,
        )
        print(f"  Probed {len(atlas.layer_geometries)} layers")

        polish_tokens = owt_corpus.probe_holdout
        print(f"  Polish tokens: {polish_tokens.shape}")

        # ---- Per-block polish ----
        # For each block, build a single-block student, polish it, store polished W_down.
        print(f"\n  Per-block polish ({POLISH_STEPS} steps each)...")
        polished_weights: dict[int, dict] = {}

        for block_id in ALL_BLOCKS:
            print(f"    Block {block_id}...")
            _set_seed(seed)
            # Build single-block pack
            single_pack = _derive_block_pack_from_atlas(atlas, budget, [block_id])
            # Swap into fresh copy
            single_student = swap_compressed_ffns(teacher, single_pack, inplace=False)
            single_student.eval()

            # Polish
            losses = _polish_block_v2(
                single_student,
                block_id,
                polish_tokens,
                n_steps=POLISH_STEPS,
                lr=POLISH_LR,
                device=teacher.device,
            )
            if losses:
                print(f"      polish loss: {losses[0]:.4f} → {losses[-1]:.4f}")

            # Store polished weights
            polished_weights[block_id] = _extract_polished_weights(single_student, block_id)

        print(f"  Polished {len(polished_weights)} blocks.")

        # ---- Accumulation sweep ----
        for ordering_name, ordering in [("forward", FORWARD_ORDER), ("reverse", REVERSE_ORDER)]:
            print(f"\n  Accumulation sweep: {ordering_name} order {ordering}")

            # Build full atlas-based CompressedFfnPacks for all k
            # We need a full pack of all 8 blocks, then for each k we swap k blocks
            # using their polished weights.

            for k in range(1, len(ordering) + 1):
                active_blocks = ordering[:k]
                print(f"    k={k}: blocks={active_blocks}", end="")

                _set_seed(seed)

                # Build pack for active blocks (unpolished)
                k_pack = _derive_block_pack_from_atlas(atlas, budget, active_blocks)

                # Swap all k blocks into a fresh student
                k_student = swap_compressed_ffns(teacher, k_pack, inplace=False)
                k_student.eval()

                # Inject polished weights for each active block
                for block_id in active_blocks:
                    if block_id in polished_weights and polished_weights[block_id]:
                        _inject_polished_weights(k_student, block_id, polished_weights[block_id])

                # Evaluate
                metrics = _evaluate_model(teacher, k_student, owt_corpus, batch_size=BATCH_SIZE)
                ppl = metrics["perplexity"]
                delta_pct = 100.0 * (ppl - baseline_ppl) / baseline_ppl
                print(f" → PPL={ppl:.3f} ({delta_pct:+.1f}%)")

                all_results[seed][ordering_name][k] = ppl

    # ---- Aggregate over seeds ----
    print("\n" + "="*60)
    print("AGGREGATING RESULTS")
    print("="*60)

    mean_baseline = float(torch.tensor(list(baseline_ppls.values()), dtype=torch.float32).mean().item())
    std_baseline = float(torch.tensor(list(baseline_ppls.values()), dtype=torch.float32).std().item())
    print(f"Baseline PPL: {mean_baseline:.3f} ± {std_baseline:.3f}")

    agg_results: dict[str, dict] = {}

    for ordering_name in ["forward", "reverse"]:
        ordering = FORWARD_ORDER if ordering_name == "forward" else REVERSE_ORDER
        agg_results[ordering_name] = {
            "order": ordering,
            "by_k": {},
        }

        ks = list(range(1, len(ordering) + 1))
        mean_ppls = []
        std_ppls = []

        for k in ks:
            ppls_k = [all_results[s][ordering_name][k] for s in SEEDS]
            mean_ppl = float(torch.tensor(ppls_k, dtype=torch.float32).mean().item())
            std_ppl = float(torch.tensor(ppls_k, dtype=torch.float32).std().item())
            mean_delta = 100.0 * (mean_ppl - mean_baseline) / mean_baseline
            mean_ppls.append(mean_ppl)
            std_ppls.append(std_ppl)

            blocks_active = ordering[:k]
            print(f"  {ordering_name} k={k} blocks={blocks_active}: PPL={mean_ppl:.3f}±{std_ppl:.3f} ({mean_delta:+.1f}%)")

            agg_results[ordering_name]["by_k"][k] = {
                "blocks_active": blocks_active,
                "ppl_mean": mean_ppl,
                "ppl_std": std_ppl,
                "delta_pct_mean": mean_delta,
                "per_seed": {s: all_results[s][ordering_name][k] for s in SEEDS},
            }

        # Fit scaling laws using mean PPL across k=1..8
        # Include k=0 (baseline) for better fitting
        ks_fit = ks
        ppls_fit = mean_ppls

        # Additive fit
        a_add, b_add, r2_add, rmse_add = _fit_additive(ks_fit, ppls_fit)
        # Multiplicative fit
        a_mul, c_mul, r2_mul, rmse_mul = _fit_multiplicative(ks_fit, ppls_fit)
        # Power-law fit
        a_pow, b_pow, r2_pow, rmse_pow = _fit_power_law(ks_fit, ppls_fit)

        print(f"\n  {ordering_name} scaling laws:")
        print(f"    Additive:       PPL(k) = {a_add:.3f} + {b_add:.4f}*k  R²={r2_add:.4f} RMSE={rmse_add:.4f}")
        print(f"    Multiplicative: PPL(k) = {a_mul:.3f} * (1+{c_mul:.4f})^k  R²={r2_mul:.4f} RMSE={rmse_mul:.4f}")
        print(f"    Power-law:      PPL(k) = {a_pow:.3f} * (k+1)^{b_pow:.4f}  R²={r2_pow:.4f} RMSE={rmse_pow:.4f}")

        # Verdict: sub- vs super-multiplicative at k=8
        predicted_multiplicative_k8 = a_mul * (1 + c_mul) ** 8
        actual_k8 = mean_ppls[-1]
        if actual_k8 < predicted_multiplicative_k8 * 1.02:
            regime = "sub_multiplicative"
        elif actual_k8 > predicted_multiplicative_k8 * 1.05:
            regime = "super_multiplicative"
        else:
            regime = "approximately_multiplicative"

        # Sub-multiplicative check: compare to c_mul vs single-block avg delta
        single_block_delta = (mean_ppls[0] - mean_baseline) / mean_baseline
        if c_mul < single_block_delta and r2_mul > 0.9:
            verdict = "CONFIRMED_SUB_MULTIPLICATIVE"
        elif regime == "super_multiplicative":
            verdict = "REFUTED_SUPER_MULTIPLICATIVE"
        else:
            verdict = "INCONCLUSIVE"

        agg_results[ordering_name]["scaling_laws"] = {
            "additive": {"a": a_add, "b": b_add, "r2": r2_add, "rmse": rmse_add},
            "multiplicative": {"a": a_mul, "c": c_mul, "r2": r2_mul, "rmse": rmse_mul},
            "power_law": {"a": a_pow, "b": b_pow, "r2": r2_pow, "rmse": rmse_pow},
        }
        agg_results[ordering_name]["regime"] = regime
        agg_results[ordering_name]["verdict"] = verdict
        agg_results[ordering_name]["mean_ppls"] = mean_ppls
        agg_results[ordering_name]["std_ppls"] = std_ppls

    # ---- Plot ----
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("GPT-2 FFN: Compositional Error Accumulation (k=1..8 blocks, 6× polished)", fontsize=13)

    ks = list(range(1, 9))
    colors = {"forward": "#2196F3", "reverse": "#FF5722"}
    labels = {"forward": "Forward (least-damage-first)", "reverse": "Reverse (most-damage-first)"}

    ax = axes[0]
    for ordering_name in ["forward", "reverse"]:
        res = agg_results[ordering_name]
        mean_ppls = res["mean_ppls"]
        std_ppls = res["std_ppls"]
        c = colors[ordering_name]

        # Plot mean ± std
        ax.plot(ks, mean_ppls, "o-", color=c, label=labels[ordering_name], linewidth=2, markersize=6)
        ax.fill_between(ks,
                        [m - s for m, s in zip(mean_ppls, std_ppls)],
                        [m + s for m, s in zip(mean_ppls, std_ppls)],
                        alpha=0.2, color=c)

        # Overlay additive fit
        sl = res["scaling_laws"]
        ks_fine = np.linspace(1, 8, 100)
        add_fit = sl["additive"]["a"] + sl["additive"]["b"] * ks_fine
        mul_fit = sl["multiplicative"]["a"] * (1 + sl["multiplicative"]["c"]) ** ks_fine
        ls_add = "--" if ordering_name == "forward" else "-."
        ls_mul = ":" if ordering_name == "forward" else (0, (3, 1, 1, 1))
        ax.plot(ks_fine, add_fit, ls_add, color=c, alpha=0.5, linewidth=1.2,
                label=f"{ordering_name} additive R²={sl['additive']['r2']:.3f}")
        ax.plot(ks_fine, mul_fit, ":", color=c, alpha=0.7, linewidth=1.5,
                label=f"{ordering_name} mult. R²={sl['multiplicative']['r2']:.3f}")

    ax.axhline(mean_baseline, color="gray", linestyle="--", linewidth=1, label=f"Baseline {mean_baseline:.2f}")
    ax.set_xlabel("k (number of compressed blocks)")
    ax.set_ylabel("Perplexity")
    ax.set_title("PPL vs k with scaling law fits")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)

    # Panel (b): per-k residual from multiplicative fit
    ax2 = axes[1]
    for ordering_name in ["forward", "reverse"]:
        res = agg_results[ordering_name]
        mean_ppls = res["mean_ppls"]
        sl = res["scaling_laws"]
        c = colors[ordering_name]
        mul_pred = [sl["multiplicative"]["a"] * (1 + sl["multiplicative"]["c"]) ** k for k in ks]
        residuals = [100.0 * (actual - pred) / pred for actual, pred in zip(mean_ppls, mul_pred)]
        ax2.plot(ks, residuals, "o-", color=c, label=labels[ordering_name], linewidth=2, markersize=6)

    ax2.axhline(0, color="gray", linestyle="--", linewidth=1, label="Multiplicative baseline")
    ax2.set_xlabel("k (number of compressed blocks)")
    ax2.set_ylabel("Residual from multiplicative fit (%)")
    ax2.set_title("Super/sub-multiplicative residual\n(positive = super-multiplicative)")
    ax2.legend(fontsize=9)
    ax2.set_xticks(ks)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_089_accumulation_curve.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    # ---- Save JSON ----
    results = {
        "experiment": "cycle_089_compositional_accumulation",
        "item": 104,
        "forward_order": FORWARD_ORDER,
        "reverse_order": REVERSE_ORDER,
        "hidden_fraction": HIDDEN_FRACTION,
        "polish_steps": POLISH_STEPS,
        "polish_lr": POLISH_LR,
        "seeds": SEEDS,
        "baseline_ppl_mean": mean_baseline,
        "baseline_ppl_std": std_baseline,
        "baseline_ppl_per_seed": baseline_ppls,
        "orderings": agg_results,
        "per_seed_raw": {
            str(s): {
                "forward": {str(k): all_results[s]["forward"][k] for k in range(1, 9)},
                "reverse": {str(k): all_results[s]["reverse"][k] for k in range(1, 9)},
                "baseline_ppl": baseline_ppls[s],
            }
            for s in SEEDS
        },
    }

    json_path = RESULTS_DIR / "cycle_089_accumulation_curve.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved JSON: {json_path}")

    # ---- Summary ----
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline PPL (OWT): {mean_baseline:.3f} ± {std_baseline:.3f}")
    for ordering_name in ["forward", "reverse"]:
        res = agg_results[ordering_name]
        k8_ppl = res["by_k"][8]["ppl_mean"]
        k8_delta = res["by_k"][8]["delta_pct_mean"]
        sl = res["scaling_laws"]
        print(f"\n{ordering_name.upper()} ordering:")
        print(f"  k=8 PPL: {k8_ppl:.3f} ({k8_delta:+.1f}%)")
        print(f"  Additive R²={sl['additive']['r2']:.4f}  b={sl['additive']['b']:.4f}")
        print(f"  Multiplicative R²={sl['multiplicative']['r2']:.4f}  c={sl['multiplicative']['c']:.4f}")
        print(f"  Power-law R²={sl['power_law']['r2']:.4f}  exponent={sl['power_law']['b']:.4f}")
        print(f"  Regime: {res['regime']}")
        print(f"  Verdict: {res['verdict']}")

    return results


if __name__ == "__main__":
    run_experiment()
