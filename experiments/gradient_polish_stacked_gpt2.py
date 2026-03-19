"""Cycle 059: Gradient-polished lstsq on stacked blocks — GPT-2 FFN.

Tests whether Adam steps after analytic lstsq refit reduce the PPL degradation
that arises from stacking k=4 compressed blocks at 6×.

Baseline: pure lstsq stacking (0 polish steps) = ~+30% PPL (cycle 058).
Polish variants: {10, 25, 50} Adam steps at lr=1e-4 on W_down columns only.
Polish data: probe_holdout (4096 tokens, disjoint from lstsq probe_train and eval_val).

Criterion: 50 steps reduce stacked PPL degradation by >=30% relative
           (e.g., +30% → ≤+21%).
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
PROBE_TRAIN_SEQS = 200          # For lstsq geometry probing
PROBE_HOLDOUT_SEQS = 200        # Becomes polish data (4096 tokens ~ 32 seqs x 128)
EVAL_VAL_SEQS = 200
SEQUENCE_LENGTH = 128
BATCH_SIZE = 4

POLISH_STEPS_LIST = [0, 10, 25, 50]  # 0 = pure lstsq baseline
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


def _derive_single_block_pack(
    atlas: FfnGeometryAtlas,
    budget: CompressionBudgetSpec,
    block_id: int,
) -> CompressedFfnPack:
    """Build a CompressedFfnPack for a single block (returns pack for that block only)."""
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


def _find_compressed_mlp(model: nn.Module, block_id: int):
    """Return the CompressedMLP module for a given block_id, or None."""
    try:
        block = model.transformer.h[block_id]
    except (AttributeError, IndexError):
        return None
    # CompressedMLP replaces the mlp submodule
    mlp = getattr(block, "mlp", None)
    return mlp


def _polish_block(
    student_model: nn.Module,
    block_id: int,
    polish_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """Run n_steps of Adam on W_down (c_proj.weight) of the given compressed block.

    Freezes all params except that layer's c_proj.weight.
    Returns list of per-step losses.
    """
    if n_steps == 0:
        return []

    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        print(f"  WARNING: no c_proj found for block {block_id}, skipping polish")
        return []

    # Freeze everything
    student_model.requires_grad_(False)

    # Unfreeze only the target W_down
    w_down = mlp.c_proj.weight  # shape: [out_dim, compressed_hidden]
    w_down.requires_grad_(True)
    # Also unfreeze bias if present
    b_down = getattr(mlp.c_proj, "bias", None)
    if b_down is not None:
        b_down.requires_grad_(True)

    params_to_optimize = [w_down] + ([b_down] if b_down is not None else [])
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

    student_model.train()
    losses = []
    # Shuffle polish tokens for each step to avoid ordering bias
    n_seqs = polish_tokens.shape[0]
    for step in range(n_steps):
        # Random mini-batch from polish tokens
        idx = torch.randperm(n_seqs, generator=torch.Generator().manual_seed(step))[:BATCH_SIZE]
        input_ids = polish_tokens[idx].to(device)
        targets = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()

        optimizer.zero_grad()
        logits = student_model(inputs).logits if hasattr(student_model(inputs), "logits") else student_model(inputs)[0]
        # GPT-2 returns a tuple; logits are index 0
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    student_model.eval()
    # Freeze back
    student_model.requires_grad_(False)
    return losses


def _forward_for_polish(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Run model and return logits, handling GPT-2 tuple output."""
    out = model(input_ids)
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "logits"):
        return out.logits
    return out


def _polish_block_v2(
    student_model: nn.Module,
    block_id: int,
    polish_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """Run n_steps of Adam on W_down (c_proj.weight) of the given compressed block.

    Uses a cleaner forward pass that avoids repeated model() calls.
    """
    if n_steps == 0:
        return []

    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        print(f"  WARNING: no c_proj found for block {block_id}, skipping polish")
        return []

    # Freeze everything
    student_model.requires_grad_(False)

    # Unfreeze only W_down (c_proj weight + optional bias)
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
        # Random mini-batch
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = polish_tokens[idx].to(device)          # [B, seq_len]
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
    # Re-freeze all
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

    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        min_hidden=32,
        selection_mode="topk",
    )

    # Results storage: polish_ppls[seed][n_steps] = ppl
    polish_ppls: dict[int, dict[int, float]] = {s: {} for s in SEEDS}
    polish_losses: dict[int, dict[int, list[list[float]]]] = {s: {} for s in SEEDS}  # [seed][n_steps][block_idx]

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

        # Probe geometry for all 4 target blocks at once (uses probe_train data)
        print(f"\n  -- Probing FFN geometry for blocks {TARGET_BLOCKS} --")
        _set_seed(seed)
        atlas = probe_ffn_geometry(
            teacher,
            owt_corpus,
            layer_ids=tuple(TARGET_BLOCKS),
            max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
            batch_size=BATCH_SIZE,
        )

        # Polish data: probe_holdout (disjoint from lstsq probe_train and eval_val)
        polish_tokens = owt_corpus.probe_holdout  # [n_seqs, seq_len]
        print(f"  Polish tokens: {polish_tokens.shape} = {polish_tokens.numel()} tokens")

        for n_steps in POLISH_STEPS_LIST:
            print(f"\n  -- Polish steps = {n_steps} --")
            _set_seed(seed)

            # Build fresh student from scratch (pure lstsq)
            pack_all = derive_ffn_compression(atlas, budget)
            student = swap_compressed_ffns(teacher, pack_all, inplace=False)
            student.eval()

            if n_steps > 0:
                step_losses_per_block = []
                for block_id in TARGET_BLOCKS:
                    print(f"    Polishing block {block_id} for {n_steps} steps...")
                    losses = _polish_block_v2(
                        student,
                        block_id,
                        polish_tokens,
                        n_steps=n_steps,
                        lr=POLISH_LR,
                        device=teacher.device,
                    )
                    first_loss = losses[0] if losses else float("nan")
                    last_loss = losses[-1] if losses else float("nan")
                    print(f"      loss: {first_loss:.4f} → {last_loss:.4f}")
                    step_losses_per_block.append(losses)
                polish_losses[seed][n_steps] = step_losses_per_block
            else:
                polish_losses[seed][n_steps] = []

            # Evaluate compressed student
            metrics = _evaluate_model(teacher, student, owt_corpus, batch_size=BATCH_SIZE)
            ppl = metrics["perplexity"]
            polish_ppls[seed][n_steps] = ppl
            delta_pct = 100.0 * (ppl - baseline_ppl) / baseline_ppl
            print(f"  steps={n_steps}: PPL={ppl:.3f} ({delta_pct:+.1f}%)")

    # ---- Aggregate over seeds ----
    def agg(vals: list[float]) -> tuple[float, float]:
        t = torch.tensor(vals, dtype=torch.float32)
        return float(t.mean().item()), float(t.std().item()) if len(vals) > 1 else 0.0

    polish_agg: dict[int, dict] = {}
    for n_steps in POLISH_STEPS_LIST:
        ppls = [polish_ppls[s][n_steps] for s in SEEDS]
        mean_ppl, std_ppl = agg(ppls)
        deltas = [100.0 * (p - baseline_ppl) / baseline_ppl for p in ppls]
        mean_delta, std_delta = agg(deltas)
        polish_agg[n_steps] = {
            "ppl_mean": mean_ppl,
            "ppl_std": std_ppl,
            "delta_pct_mean": mean_delta,
            "delta_pct_std": std_delta,
            "per_seed": {s: polish_ppls[s][n_steps] for s in SEEDS},
        }

    # Relative improvement vs 0-step baseline
    baseline_delta = polish_agg[0]["delta_pct_mean"]
    for n_steps in POLISH_STEPS_LIST:
        if abs(baseline_delta) > 1e-6:
            rel_improvement = 100.0 * (baseline_delta - polish_agg[n_steps]["delta_pct_mean"]) / abs(baseline_delta)
        else:
            rel_improvement = float("nan")
        polish_agg[n_steps]["relative_improvement_pct"] = rel_improvement

    # Verdict
    rel_50 = polish_agg[50]["relative_improvement_pct"]
    if rel_50 >= 30.0:
        verdict = "CONFIRMED"
    elif polish_agg[50]["delta_pct_mean"] > baseline_delta:
        verdict = "REFUTED_OVERFITTING"
    elif rel_50 > 0 and rel_50 < 30.0:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "REFUTED_NO_IMPROVEMENT"

    results = {
        "baseline_ppl": baseline_ppl,
        "target_blocks": TARGET_BLOCKS,
        "hidden_fraction": HIDDEN_FRACTION,
        "seeds": SEEDS,
        "polish_steps": POLISH_STEPS_LIST,
        "polish_lr": POLISH_LR,
        "polish_agg": polish_agg,
        "verdict": verdict,
    }

    return results


def make_plot(results: dict, out_path: Path) -> None:
    steps = results["polish_steps"]
    baseline_ppl = results["baseline_ppl"]

    delta_means = [results["polish_agg"][n]["delta_pct_mean"] for n in steps]
    delta_stds = [results["polish_agg"][n]["delta_pct_std"] for n in steps]
    rel_impr = [results["polish_agg"][n]["relative_improvement_pct"] for n in steps]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Gradient-Polished lstsq — GPT-2 FFN k=4@6× (OWT)\n"
                 f"Baseline PPL={baseline_ppl:.3f}", fontsize=12)

    # Left: PPL delta vs steps
    ax = axes[0]
    colors = ["#2196F3", "#FF9800", "#E53935", "#9C27B0"]
    bars = ax.bar(range(len(steps)), delta_means, 0.55, yerr=delta_stds,
                  color=colors[:len(steps)], alpha=0.85, capsize=5)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([f"{n} steps" for n in steps])
    ax.set_xlabel("Polish steps (Adam)")
    ax.set_ylabel("PPL % change vs baseline")
    ax.set_title(f"PPL degradation after polish\n(verdict: {results['verdict']})")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axhline(delta_means[0] * 0.7, color="green", lw=1.5, ls="--",
               label="30% relative target")
    ax.legend(fontsize=9)

    # Add text labels
    for i, (m, s) in enumerate(zip(delta_means, delta_stds)):
        ax.text(i, m + s + 0.3, f"{m:+.1f}%", ha="center", fontsize=9)

    # Right: relative improvement %
    ax2 = axes[1]
    valid_steps = [n for n in steps if n > 0]
    valid_rel = [results["polish_agg"][n]["relative_improvement_pct"] for n in valid_steps]

    ax2.plot(valid_steps, valid_rel, "o-", color="#4CAF50", linewidth=2, markersize=8)
    ax2.axhline(30.0, color="red", lw=1.5, ls="--", label="30% criterion")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("Polish steps (Adam)")
    ax2.set_ylabel("Relative improvement vs 0-step baseline (%)")
    ax2.set_title("Relative PPL degradation reduction")
    ax2.legend(fontsize=9)

    for i, (s, r) in enumerate(zip(valid_steps, valid_rel)):
        ax2.text(s, r + 1.0, f"{r:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved plot to {out_path}")


def main() -> None:
    print("Cycle 059: Gradient-polished lstsq — GPT-2 FFN (k=4@6×, OWT)")
    results = run_experiment()

    json_path = RESULTS_DIR / "cycle_059_gradient_polish.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved results to {json_path}")

    png_path = RESULTS_DIR / "cycle_059_gradient_polish.png"
    make_plot(results, png_path)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Baseline GPT-2 PPL (OWT): {results['baseline_ppl']:.3f}")
    print(f"Target blocks (k=4): {results['target_blocks']}")
    print(f"Compression ratio: {1 / results['hidden_fraction']:.0f}×")
    print(f"Polish LR: {results['polish_lr']}")

    print(f"\n{'Steps':<10} {'PPL mean':<12} {'Delta%':<12} {'Std%':<10} {'Rel.Impr%':<14}")
    print("-" * 60)
    for n in results["polish_steps"]:
        d = results["polish_agg"][n]
        ri = d["relative_improvement_pct"]
        ri_str = f"{ri:.1f}%" if not math.isnan(ri) else "baseline"
        print(f"{n:<10} {d['ppl_mean']:<12.3f} {d['delta_pct_mean']:+.2f}%{' ':<5} "
              f"{d['delta_pct_std']:.2f}%{' ':<4} {ri_str}")

    print(f"\n*** VERDICT: {results['verdict']} ***")
    criterion_met = results["verdict"] == "CONFIRMED"
    print(f"Criterion (≥30% relative improvement at 50 steps): {'MET' if criterion_met else 'NOT MET'}")


if __name__ == "__main__":
    main()
