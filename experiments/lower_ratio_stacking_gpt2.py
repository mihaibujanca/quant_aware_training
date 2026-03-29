"""Cycle 093: Lower compression ratio stacking — GPT-2 FFN.

For 3 compression ratios (3×, 4×, 6×) and k=1..8 compressed blocks,
measures how PPL degrades as more blocks are stacked simultaneously.
Forward (least-damage-first) ordering only.
Fits multiplicative model PPL(k) = a * (1+c)^k per ratio.

Hypothesis: c decreases as compression ratio decreases (lower compression = less error per block).
Confirmed if c_3x <= 0.20 AND k=8 PPL at 3x <= +20%.
Refuted if c_3x >= 0.30.
Sanity check: 6x should reproduce c~0.41 from cycle 089.
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

# Three compression ratios: 3x (k=1024), 4x (k=768), 6x (k=512) from 3072 hidden
RATIOS = {
    "3x": 1 / 3,
    "4x": 1 / 4,
    "6x": 1 / 6,
}

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

def _fit_multiplicative(ks: list[int], ppls: list[float]) -> tuple[float, float, float, float]:
    """Fit PPL(k) = a * (1+c)^k via log-linearization. Returns (a, c, r2, rmse)."""
    ks_t = torch.tensor(ks, dtype=torch.float64)
    ppls_t = torch.tensor(ppls, dtype=torch.float64)
    log_ppls = torch.log(ppls_t)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment() -> dict:
    print("Loading GPT-2 teacher...")
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2", device="cpu")
    )
    print(f"Teacher loaded: {teacher.model_name}, device={teacher.device}")

    ALL_BLOCKS = list(set(FORWARD_ORDER))

    # Results structure: [ratio_name][seed][k] -> ppl
    all_results: dict[str, dict[int, dict[int, float]]] = {
        rn: {s: {} for s in SEEDS} for rn in RATIOS
    }
    baseline_ppls: dict[int, float] = {}

    # Single-block deltas per ratio (for advisory logging)
    single_block_deltas: dict[str, dict[int, float]] = {rn: {} for rn in RATIOS}
    # Final polish losses per ratio per block
    polish_losses_final: dict[str, dict[int, float]] = {rn: {} for rn in RATIOS}

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

        # Baseline PPL (once per seed, shared across ratios)
        if seed not in baseline_ppls:
            bl_metrics = _evaluate_model(teacher, teacher.model, owt_corpus, batch_size=BATCH_SIZE)
            baseline_ppls[seed] = bl_metrics["perplexity"]
        baseline_ppl = baseline_ppls[seed]
        print(f"  Baseline PPL (OWT): {baseline_ppl:.3f}")

        # Probe geometry ONCE per seed (ratio-independent)
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

        # Per-ratio loop
        for ratio_name, hidden_frac in RATIOS.items():
            print(f"\n  --- Ratio {ratio_name} (hidden_fraction={hidden_frac:.4f}) ---")

            budget = CompressionBudgetSpec(
                hidden_fraction=hidden_frac,
                min_hidden=32,
                selection_mode="topk",
            )

            # ---- Per-block polish for this ratio ----
            print(f"  Per-block polish ({POLISH_STEPS} steps each)...")
            polished_weights: dict[int, dict] = {}

            for block_id in ALL_BLOCKS:
                print(f"    Block {block_id}...", end="")
                _set_seed(seed)
                single_pack = _derive_block_pack_from_atlas(atlas, budget, [block_id])
                single_student = swap_compressed_ffns(teacher, single_pack, inplace=False)
                single_student.eval()

                losses = _polish_block_v2(
                    single_student,
                    block_id,
                    polish_tokens,
                    n_steps=POLISH_STEPS,
                    lr=POLISH_LR,
                    device=teacher.device,
                )
                final_loss = losses[-1] if losses else float("nan")
                print(f" polish: {losses[0]:.4f} → {final_loss:.4f}" if losses else " (no polish)")

                # Log final polish loss
                if ratio_name not in polish_losses_final:
                    polish_losses_final[ratio_name] = {}
                polish_losses_final[ratio_name][block_id] = final_loss

                polished_weights[block_id] = _extract_polished_weights(single_student, block_id)

                # Log single-block delta
                metrics_1 = _evaluate_model(teacher, single_student, owt_corpus, batch_size=BATCH_SIZE)
                ppl_1 = metrics_1["perplexity"]
                delta_1 = 100.0 * (ppl_1 - baseline_ppl) / baseline_ppl
                print(f"      Single-block PPL={ppl_1:.3f} ({delta_1:+.1f}%)")
                if block_id not in single_block_deltas[ratio_name]:
                    single_block_deltas[ratio_name][block_id] = []
                single_block_deltas[ratio_name][block_id].append(delta_1)

            # ---- Accumulation sweep ----
            print(f"\n  Accumulation sweep: forward order {FORWARD_ORDER}")
            for k in range(1, len(FORWARD_ORDER) + 1):
                active_blocks = FORWARD_ORDER[:k]
                print(f"    k={k}: blocks={active_blocks}", end="")

                _set_seed(seed)
                k_pack = _derive_block_pack_from_atlas(atlas, budget, active_blocks)
                k_student = swap_compressed_ffns(teacher, k_pack, inplace=False)
                k_student.eval()

                for block_id in active_blocks:
                    if block_id in polished_weights and polished_weights[block_id]:
                        _inject_polished_weights(k_student, block_id, polished_weights[block_id])

                metrics = _evaluate_model(teacher, k_student, owt_corpus, batch_size=BATCH_SIZE)
                ppl = metrics["perplexity"]
                delta_pct = 100.0 * (ppl - baseline_ppl) / baseline_ppl
                print(f" → PPL={ppl:.3f} ({delta_pct:+.1f}%)")

                all_results[ratio_name][seed][k] = ppl

                # Abort check
                if ppl > 5000:
                    print(f"  ABORT: PPL {ppl:.1f} > 5000 at k={k}, ratio={ratio_name}. Stopping this ratio.")
                    for kk in range(k + 1, len(FORWARD_ORDER) + 1):
                        all_results[ratio_name][seed][kk] = float("nan")
                    break

    # ---- Aggregate over seeds ----
    print("\n" + "="*60)
    print("AGGREGATING RESULTS")
    print("="*60)

    mean_baseline = float(torch.tensor(list(baseline_ppls.values()), dtype=torch.float32).mean().item())
    std_baseline = float(torch.tensor(list(baseline_ppls.values()), dtype=torch.float32).std().item())
    print(f"Baseline PPL: {mean_baseline:.3f} ± {std_baseline:.3f}")

    if mean_baseline < 40 or mean_baseline > 65:
        print(f"  WARNING: Baseline PPL {mean_baseline:.3f} outside [40, 65] — possible data issue!")

    agg_results: dict[str, dict] = {}
    ks = list(range(1, len(FORWARD_ORDER) + 1))

    print("\n\nRATIO | k | PPL | delta%")
    print("-" * 50)

    for ratio_name in RATIOS:
        agg_results[ratio_name] = {"by_k": {}}
        mean_ppls = []
        std_ppls = []

        for k in ks:
            ppls_k = [all_results[ratio_name][s][k] for s in SEEDS]
            ppls_k_valid = [p for p in ppls_k if not math.isnan(p)]
            if ppls_k_valid:
                mean_ppl = float(torch.tensor(ppls_k_valid, dtype=torch.float32).mean().item())
                std_ppl = float(torch.tensor(ppls_k_valid, dtype=torch.float32).std().item()) if len(ppls_k_valid) > 1 else 0.0
            else:
                mean_ppl = float("nan")
                std_ppl = float("nan")
            mean_delta = 100.0 * (mean_ppl - mean_baseline) / mean_baseline if not math.isnan(mean_ppl) else float("nan")
            mean_ppls.append(mean_ppl)
            std_ppls.append(std_ppl)

            print(f"  {ratio_name}  k={k}  PPL={mean_ppl:.3f}±{std_ppl:.3f}  ({mean_delta:+.1f}%)")

            agg_results[ratio_name]["by_k"][k] = {
                "blocks_active": FORWARD_ORDER[:k],
                "ppl_mean": mean_ppl,
                "ppl_std": std_ppl,
                "delta_pct_mean": mean_delta,
                "per_seed": {s: all_results[ratio_name][s][k] for s in SEEDS},
            }

        # Filter out NaN for fitting
        valid_pairs = [(k, p) for k, p in zip(ks, mean_ppls) if not math.isnan(p)]
        if len(valid_pairs) >= 2:
            fit_ks = [kk for kk, _ in valid_pairs]
            fit_ppls = [pp for _, pp in valid_pairs]
            a_mul, c_mul, r2_mul, rmse_mul = _fit_multiplicative(fit_ks, fit_ppls)
        else:
            a_mul, c_mul, r2_mul, rmse_mul = float("nan"), float("nan"), float("nan"), float("nan")

        k4_delta = agg_results[ratio_name]["by_k"][4]["delta_pct_mean"] if 4 in agg_results[ratio_name]["by_k"] else float("nan")
        k8_delta = agg_results[ratio_name]["by_k"][8]["delta_pct_mean"] if 8 in agg_results[ratio_name]["by_k"] else float("nan")

        print(f"\n  {ratio_name} multiplicative fit: a={a_mul:.3f}, c={c_mul:.4f}, R²={r2_mul:.4f}, RMSE={rmse_mul:.4f}")
        print(f"  {ratio_name} k=4 delta: {k4_delta:+.1f}%  k=8 delta: {k8_delta:+.1f}%")

        # Single-block deltas summary
        for block_id in sorted(single_block_deltas[ratio_name].keys()):
            deltas = single_block_deltas[ratio_name][block_id]
            mean_d = sum(deltas) / len(deltas)
            print(f"    single-block delta block {block_id}: {mean_d:+.1f}%")

        agg_results[ratio_name]["scaling_multiplicative"] = {
            "a": a_mul, "c": c_mul, "r2": r2_mul, "rmse": rmse_mul
        }
        agg_results[ratio_name]["mean_ppls"] = mean_ppls
        agg_results[ratio_name]["std_ppls"] = std_ppls
        agg_results[ratio_name]["k4_delta_pct"] = k4_delta
        agg_results[ratio_name]["k8_delta_pct"] = k8_delta

    # ---- Verdict ----
    c_3x = agg_results["3x"]["scaling_multiplicative"]["c"]
    k8_delta_3x = agg_results["3x"]["k8_delta_pct"]

    if c_3x <= 0.20 and k8_delta_3x <= 20.0:
        verdict = "CONFIRMED"
    elif c_3x >= 0.30:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"  c_3x={c_3x:.4f}, k8_delta_3x={k8_delta_3x:.1f}%")
    print(f"  VERDICT: {verdict}")

    # Monotonicity check
    c_4x = agg_results["4x"]["scaling_multiplicative"]["c"]
    c_6x = agg_results["6x"]["scaling_multiplicative"]["c"]
    print(f"\nMonotonicity check: c_3x={c_3x:.4f} c_4x={c_4x:.4f} c_6x={c_6x:.4f}")
    if c_3x <= c_4x <= c_6x:
        print("  Monotonicity: SATISFIED (c_3x <= c_4x <= c_6x)")
    else:
        print("  Monotonicity: VIOLATED — inspect for bugs!")

    # 6x sanity check
    print(f"\n6x sanity check: c={c_6x:.4f} (expected ~0.41 from cycle 089)")

    # ---- Plot ----
    print("\nGenerating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle("GPT-2 FFN: Stacking Across Compression Ratios (k=1..8, forward order)", fontsize=13)

    colors = {"3x": "#2196F3", "4x": "#4CAF50", "6x": "#FF5722"}
    ks_arr = np.array(ks)

    for ratio_name in RATIOS:
        res = agg_results[ratio_name]
        mean_ppls = [p for p in res["mean_ppls"]]
        std_ppls = [s for s in res["std_ppls"]]
        c = colors[ratio_name]
        c_val = res["scaling_multiplicative"]["c"]
        r2_val = res["scaling_multiplicative"]["r2"]

        valid_mask = [not math.isnan(p) for p in mean_ppls]
        valid_ks = [k for k, v in zip(ks, valid_mask) if v]
        valid_ppls = [p for p, v in zip(mean_ppls, valid_mask) if v]
        valid_stds = [s for s, v in zip(std_ppls, valid_mask) if v]

        ax.plot(valid_ks, valid_ppls, "o-", color=c,
                label=f"{ratio_name} (c={c_val:.3f}, R²={r2_val:.3f})",
                linewidth=2, markersize=6)
        ax.fill_between(valid_ks,
                        [m - s for m, s in zip(valid_ppls, valid_stds)],
                        [m + s for m, s in zip(valid_ppls, valid_stds)],
                        alpha=0.15, color=c)

        # Overlay multiplicative fit
        if not math.isnan(c_val):
            a_val = res["scaling_multiplicative"]["a"]
            ks_fine = np.linspace(1, 8, 100)
            mul_fit = a_val * (1 + c_val) ** ks_fine
            ax.plot(ks_fine, mul_fit, ":", color=c, alpha=0.7, linewidth=1.5)

    ax.axhline(mean_baseline, color="gray", linestyle="--", linewidth=1.5,
               label=f"Baseline {mean_baseline:.2f}")
    ax.axhline(mean_baseline * 1.20, color="red", linestyle=":", linewidth=1,
               label=f"+20% threshold ({mean_baseline * 1.20:.2f})")

    ax.set_xlabel("k (number of compressed blocks)")
    ax.set_ylabel("Perplexity")
    ax.set_title("PPL vs k: three compression ratios")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_093_lower_ratio_stacking.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    # ---- Save JSON ----
    results = {
        "experiment": "cycle_093_lower_ratio_stacking",
        "item": 108,
        "forward_order": FORWARD_ORDER,
        "ratios": {rn: hf for rn, hf in RATIOS.items()},
        "polish_steps": POLISH_STEPS,
        "polish_lr": POLISH_LR,
        "seeds": SEEDS,
        "baseline_ppl_mean": mean_baseline,
        "baseline_ppl_std": std_baseline,
        "baseline_ppl_per_seed": baseline_ppls,
        "verdict": verdict,
        "c_3x": c_3x,
        "c_4x": c_4x,
        "c_6x": c_6x,
        "k8_delta_pct_3x": k8_delta_3x,
        "k8_delta_pct_4x": agg_results["4x"]["k8_delta_pct"],
        "k8_delta_pct_6x": agg_results["6x"]["k8_delta_pct"],
        "ratios_detail": agg_results,
        "single_block_deltas": {
            rn: {str(bid): vals for bid, vals in bdata.items()}
            for rn, bdata in single_block_deltas.items()
        },
        "polish_losses_final": {
            rn: {str(bid): v for bid, v in bdata.items()}
            for rn, bdata in polish_losses_final.items()
        },
        "per_seed_raw": {
            str(s): {
                rn: {str(k): all_results[rn][s][k] for k in ks}
                for rn in RATIOS
            }
            for s in SEEDS
        },
    }

    json_path = RESULTS_DIR / "cycle_093_lower_ratio_stacking.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved JSON: {json_path}")

    # ---- Summary table ----
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print(f"{'Ratio':<8} {'c':>8} {'R²':>8} {'k=4 Δ%':>10} {'k=8 Δ%':>10}")
    print("-" * 50)
    for ratio_name in ["3x", "4x", "6x"]:
        sl = agg_results[ratio_name]["scaling_multiplicative"]
        k4d = agg_results[ratio_name]["k4_delta_pct"]
        k8d = agg_results[ratio_name]["k8_delta_pct"]
        print(f"  {ratio_name:<6} {sl['c']:>8.4f} {sl['r2']:>8.4f} {k4d:>+10.1f} {k8d:>+10.1f}")
    print(f"\nVERDICT: {verdict}")
    print(f"(c_3x={c_3x:.4f}, k=8 delta 3x={k8_delta_3x:.1f}%)")

    return results


if __name__ == "__main__":
    run_experiment()
