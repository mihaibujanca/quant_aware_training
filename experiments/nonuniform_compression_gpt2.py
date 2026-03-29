"""Cycle 094: Non-uniform compression ratio allocation — GPT-2 FFN.

Tests whether allocating more channels to high-damage blocks (and fewer to
low-damage blocks) at matched total parameter budget improves PPL over uniform 6×.

Three allocations at TOTAL_BUDGET=2048 channels across 4 blocks [7,5,4,1]:
  - uniform_6x:          {7:512, 5:512, 4:512, 1:512}   (6× each)
  - mild_nonunif:        {7:410, 5:512, 4:512, 1:614}   (~7.5× block7, ~5× block1)
  - aggressive_nonunif:  {7:308, 5:358, 4:614, 1:768}   (10× block7, ~4× block1)

Damage order from cycle 089: block7 = least damage, block1 = most damage.
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
BLOCKS = [7, 5, 4, 1]   # least-damage to most-damage (from cycle 089)
SEEDS = [42, 123, 7]
POLISH_STEPS = 100   # reduced from 500 for runtime; all allocations use same steps (fair comparison)
POLISH_LR = 1e-4

TOTAL_BUDGET = 2048   # 4 blocks × 512 channels at 6× = 2048

# Allocation: give MORE channels to high-damage blocks (block1 > block4 > block5 > block7).
# All sums must equal TOTAL_BUDGET = 2048.
ALLOCATIONS = {
    "uniform_6x":           {7: 512, 5: 512, 4: 512, 1: 512},
    "mild_nonunif":         {7: 410, 5: 512, 4: 512, 1: 614},
    "aggressive_nonunif":   {7: 308, 5: 358, 4: 614, 1: 768},
}

# Verify budget sums
for alloc_name, alloc in ALLOCATIONS.items():
    total = sum(alloc[b] for b in BLOCKS)
    assert total == TOTAL_BUDGET, f"{alloc_name} sums to {total}, expected {TOTAL_BUDGET}"

MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200
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


def _polish_block(
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
# Main
# ---------------------------------------------------------------------------

def run_experiment() -> dict:
    print("Loading GPT-2 teacher...")
    teacher = load_nanochat_teacher(
        TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2", device="cpu")
    )
    print(f"Teacher loaded: {teacher.model_name}, device={teacher.device}")

    # Results: [alloc_name][seed] -> ppl
    all_results: dict[str, dict[int, float]] = {name: {} for name in ALLOCATIONS}
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

        # Abort condition
        if not (40.0 <= baseline_ppl <= 65.0):
            print(f"  ABORT: Baseline PPL {baseline_ppl:.3f} outside [40, 65]. Aborting seed.")
            raise RuntimeError(f"Baseline PPL {baseline_ppl:.3f} outside expected range")

        # Probe geometry for all 4 blocks at once (atlas reused across allocations)
        print(f"\n  Probing FFN geometry for blocks: {BLOCKS}")
        _set_seed(seed)
        atlas = probe_ffn_geometry(
            teacher,
            owt_corpus,
            layer_ids=tuple(sorted(BLOCKS)),
            max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
            batch_size=BATCH_SIZE,
        )
        print(f"  Probed {len(atlas.layer_geometries)} layers")

        polish_tokens = owt_corpus.probe_holdout

        # Per-allocation per-block polish
        for alloc_name, allocation in ALLOCATIONS.items():
            print(f"\n  Allocation: {alloc_name} {allocation}")
            budget = CompressionBudgetSpec(
                hidden_fraction=1 / 6,   # fallback (unused since all blocks have explicit entries)
                min_hidden=32,
                per_layer_hidden=allocation,
                selection_mode="topk",
            )

            # Polish each block independently
            polished_weights: dict[int, dict] = {}
            for block_id in BLOCKS:
                print(f"    Block {block_id} (k={allocation[block_id]})...", end=" ")
                _set_seed(seed)
                single_pack = _derive_block_pack_from_atlas(atlas, budget, [block_id])
                single_student = swap_compressed_ffns(teacher, single_pack, inplace=False)
                single_student.eval()

                losses = _polish_block(
                    single_student,
                    block_id,
                    polish_tokens,
                    n_steps=POLISH_STEPS,
                    lr=POLISH_LR,
                    device=teacher.device,
                )
                if losses:
                    print(f"polish {losses[0]:.4f} → {losses[-1]:.4f}")
                else:
                    print("no polish")

                polished_weights[block_id] = _extract_polished_weights(single_student, block_id)

            # Build full 4-block pack, swap, inject polished weights, eval
            _set_seed(seed)
            full_pack = _derive_block_pack_from_atlas(atlas, budget, BLOCKS)
            student = swap_compressed_ffns(teacher, full_pack, inplace=False)
            student.eval()

            for block_id in BLOCKS:
                if block_id in polished_weights and polished_weights[block_id]:
                    _inject_polished_weights(student, block_id, polished_weights[block_id])

            metrics = _evaluate_model(teacher, student, owt_corpus, batch_size=BATCH_SIZE)
            ppl = metrics["perplexity"]
            delta_pct = 100.0 * (ppl - baseline_ppl) / baseline_ppl
            print(f"    => PPL={ppl:.3f} ({delta_pct:+.1f}%)")

            # Abort condition: diverged
            if ppl > 500:
                print(f"  ABORT: PPL {ppl:.3f} > 500 (diverged). Aborting allocation.")
                raise RuntimeError(f"Diverged: {alloc_name} PPL={ppl:.3f}")

            all_results[alloc_name][seed] = ppl

    # ---- Aggregate over seeds ----
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)

    baseline_tensor = torch.tensor(list(baseline_ppls.values()), dtype=torch.float32)
    mean_baseline = float(baseline_tensor.mean().item())
    std_baseline = float(baseline_tensor.std().item())
    print(f"Baseline PPL: {mean_baseline:.3f} ± {std_baseline:.3f}")
    print()
    print(f"{'Allocation':<25} {'mean_ppl':>10} {'std':>8} {'delta%':>10}")
    print("-" * 60)

    agg: dict[str, dict] = {}
    for alloc_name in ALLOCATIONS:
        ppls = [all_results[alloc_name][s] for s in SEEDS]
        ppl_t = torch.tensor(ppls, dtype=torch.float32)
        mean_ppl = float(ppl_t.mean().item())
        std_ppl = float(ppl_t.std().item())
        delta_pct = 100.0 * (mean_ppl - mean_baseline) / mean_baseline
        agg[alloc_name] = {
            "mean_ppl": mean_ppl,
            "std_ppl": std_ppl,
            "delta_pct": delta_pct,
            "per_seed": {s: all_results[alloc_name][s] for s in SEEDS},
        }
        print(f"{alloc_name:<25} {mean_ppl:>10.3f} {std_ppl:>8.3f} {delta_pct:>+10.1f}%")

    # Verdict
    uniform_ppl = agg["uniform_6x"]["mean_ppl"]
    best_nonunif_delta = min(
        100.0 * (agg[name]["mean_ppl"] - uniform_ppl) / uniform_ppl
        for name in ["mild_nonunif", "aggressive_nonunif"]
    )
    best_nonunif_name = min(
        ["mild_nonunif", "aggressive_nonunif"],
        key=lambda n: agg[n]["mean_ppl"],
    )
    print()
    print(f"Best non-uniform: {best_nonunif_name}, delta vs uniform = {best_nonunif_delta:+.1f}%")

    if best_nonunif_delta <= -10.0:
        verdict = "CONFIRMED"
        verdict_note = "Non-uniform allocation achieves ≥10% relative PPL improvement over uniform 6×."
    elif best_nonunif_delta > -3.0:
        verdict = "REFUTED"
        verdict_note = "Non-uniform allocation improves PPL by <3% — allocation irrelevant after polish."
    else:
        verdict = "INCONCLUSIVE"
        verdict_note = f"Non-uniform allocation improves PPL by {abs(best_nonunif_delta):.1f}% (3-10% range)."

    print(f"VERDICT: {verdict} — {verdict_note}")

    # Sanity: uniform_6x vs cycle 089 k=4 (~80.7 PPL, ~56% delta)
    uniform_delta = agg["uniform_6x"]["delta_pct"]
    print(f"\nSanity check: uniform_6x delta={uniform_delta:+.1f}% (expect ~+56% from cycle 089)")

    # ---- Plot ----
    print("\nGenerating plot...")
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("GPT-2 FFN: Non-Uniform Compression Allocation (4 blocks, matched 2048-channel budget)", fontsize=12)

    alloc_names = list(ALLOCATIONS.keys())
    labels = {
        "uniform_6x": "Uniform 6×\n(512 each)",
        "mild_nonunif": "Mild non-uniform\n(7.5× blk7, 5× blk1)",
        "aggressive_nonunif": "Aggressive non-uniform\n(10× blk7, 4× blk1)",
    }
    colors = ["#2196F3", "#FF9800", "#E53935"]
    x = np.arange(len(alloc_names))
    bar_width = 0.5

    for i, (alloc_name, color) in enumerate(zip(alloc_names, colors)):
        mean_ppl = agg[alloc_name]["mean_ppl"]
        std_ppl = agg[alloc_name]["std_ppl"]
        ax.bar(x[i], mean_ppl, bar_width, color=color, alpha=0.85,
               yerr=std_ppl, capsize=5, label=labels[alloc_name])
        ax.text(x[i], mean_ppl + std_ppl + 0.5, f"{mean_ppl:.1f}\n({agg[alloc_name]['delta_pct']:+.1f}%)",
                ha="center", va="bottom", fontsize=9)

    ax.axhline(mean_baseline, color="gray", linestyle="--", linewidth=1.2,
               label=f"Baseline {mean_baseline:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[n] for n in alloc_names], fontsize=9)
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Verdict: {verdict} — {verdict_note}", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_094_nonuniform_compression.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    # ---- Save JSON ----
    results = {
        "experiment": "cycle_094_nonuniform_compression",
        "item": 110,
        "blocks": BLOCKS,
        "allocations": {k: {str(b): v for b, v in alloc.items()} for k, alloc in ALLOCATIONS.items()},
        "total_budget": TOTAL_BUDGET,
        "polish_steps": POLISH_STEPS,
        "polish_lr": POLISH_LR,
        "seeds": SEEDS,
        "baseline_ppl_mean": mean_baseline,
        "baseline_ppl_std": std_baseline,
        "baseline_ppl_per_seed": baseline_ppls,
        "results": agg,
        "best_nonunif_name": best_nonunif_name,
        "best_nonunif_delta_vs_uniform": best_nonunif_delta,
        "verdict": verdict,
        "verdict_note": verdict_note,
    }

    json_path = RESULTS_DIR / "cycle_094_nonuniform_compression.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved JSON: {json_path}")

    return results


if __name__ == "__main__":
    run_experiment()
