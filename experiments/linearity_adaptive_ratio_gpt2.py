"""Cycle 114: Linearity-adaptive compression ratio allocation — GPT-2 FFN (item 125).

Tests 3 ratio-allocation conditions on k=4 FFN blocks [1,4,7,11], matched budget=2048:
  (a) uniform:       all 4 blocks at k=512 (6x)
  (b) middle-heavy:  blocks 4,7 at k=384 (8x), blocks 1,11 at k=640 (4.8x)
  (c) edge-heavy:    blocks 1,11 at k=384 (8x), blocks 4,7 at k=640 (4.8x)

All: topk+lstsq + 500 Adam steps on W_down (lr=1e-4), 3 seeds, 200 eval seqs OWT.
Reports pre-polish and post-polish PPL for each condition/seed.

Verdict:
  CONFIRMED    if best non-uniform PPL improvement over uniform >= 3% relative
  REFUTED      if uniform within 1% of best non-uniform
  INCONCLUSIVE if 1-3%
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

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
SEEDS = [42, 123, 7]
EVAL_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4

FFN_BLOCKS = [1, 4, 7, 11]
TOTAL_BUDGET = 2048          # sum of k values across 4 blocks
POLISH_STEPS = 500
POLISH_LR = 1e-4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200

# Three allocation conditions — all sum to TOTAL_BUDGET = 2048
ALLOCATIONS = {
    "uniform": {1: 512, 4: 512, 7: 512, 11: 512},
    "middle_heavy": {1: 640, 4: 384, 7: 384, 11: 640},
    "edge_heavy": {1: 384, 4: 640, 7: 640, 11: 384},
}

# Verify budget constraint at import time
for alloc_name, alloc in ALLOCATIONS.items():
    total = sum(alloc.values())
    assert total == TOTAL_BUDGET, f"Budget mismatch for {alloc_name}: {total} != {TOTAL_BUDGET}"

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    if n_steps == 0:
        return []
    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        print(f"  WARNING: no c_proj for block {block_id}, skipping polish")
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
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    student_model.eval()
    student_model.requires_grad_(False)
    return losses


def _extract_polished_weights(student_model: nn.Module, block_id: int) -> dict:
    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        return {}
    return {
        "c_proj_weight": mlp.c_proj.weight.detach().clone(),
        "c_proj_bias": (
            mlp.c_proj.bias.detach().clone()
            if hasattr(mlp.c_proj, "bias") and mlp.c_proj.bias is not None
            else None
        ),
    }


def _inject_polished_weights(student_model: nn.Module, block_id: int, polished: dict) -> None:
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
print("Loading GPT-2 teacher...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()
print(f"  Teacher loaded on device={teacher.device}")

# ---------------------------------------------------------------------------
# Per-seed loop
# ---------------------------------------------------------------------------
per_seed: dict[int, dict] = {}

for seed in SEEDS:
    print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
    _set_seed(seed)

    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher,
        text_path=str(OWT_PATH),
        sequence_length=SEQ_LEN,
        probe_train_sequences=PROBE_TRAIN_SEQS,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
        eval_val_sequences=EVAL_SEQS,
        dataset_name="openwebtext",
        shuffle_seed=seed,
    ))

    # Baseline
    bl_metrics = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = float(bl_metrics["perplexity"])
    print(f"  Baseline PPL: {baseline_ppl:.3f}")

    if not (25.0 <= baseline_ppl <= 65.0):
        print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [25, 65]")
        sys.exit(1)

    # Probe FFN geometry for all blocks (once per seed)
    print(f"  Probing FFN geometry for blocks {sorted(FFN_BLOCKS)}...")
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=tuple(sorted(FFN_BLOCKS)),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )
    polish_tokens = corpus.probe_holdout

    seed_results = {"baseline": baseline_ppl}

    for alloc_name, alloc in ALLOCATIONS.items():
        print(f"\n  -- Allocation: {alloc_name} --")
        print(f"     k values: {alloc}")
        assert sum(alloc.values()) == TOTAL_BUDGET, f"Budget mismatch: {alloc}"

        budget = CompressionBudgetSpec(
            hidden_fraction=1.0 / 6.0,   # fallback (shouldn't trigger)
            min_hidden=32,
            selection_mode="topk",
            per_layer_hidden=alloc,
        )

        # Pre-polish PPL: build stacked model with all 4 blocks, no polish
        _set_seed(seed)
        pack_full = _derive_block_pack_from_atlas(atlas, budget, FFN_BLOCKS)
        student_pre = swap_compressed_ffns(teacher, pack_full, inplace=False)
        student_pre.eval()

        # Log per-block compressed param counts
        total_compressed = 0
        for layer_obj in pack_full.layers:
            total_compressed += layer_obj.compressed_param_count
            print(f"     block {layer_obj.layer_id}: k={layer_obj.compressed_hidden_dim}, "
                  f"params={layer_obj.compressed_param_count:,}")
        print(f"     total compressed params (4 blocks): {total_compressed:,}")

        pre_metrics = _evaluate_model(teacher, student_pre, corpus, batch_size=BATCH_SIZE)
        pre_ppl = float(pre_metrics["perplexity"])
        pre_delta = (pre_ppl - baseline_ppl) / baseline_ppl * 100.0
        print(f"     Pre-polish PPL: {pre_ppl:.3f} ({pre_delta:+.2f}%)")

        if pre_ppl > 500.0:
            print(f"ABORT: {alloc_name} pre-polish PPL={pre_ppl:.1f} > 500")
            sys.exit(1)

        del student_pre

        # Per-block polish (single-block mode)
        polished_weights: dict[int, dict] = {}
        for block_id in FFN_BLOCKS:
            _set_seed(seed)
            single_pack = _derive_block_pack_from_atlas(atlas, budget, [block_id])
            single_student = swap_compressed_ffns(teacher, single_pack, inplace=False)
            single_student.eval()
            losses = _polish_block(single_student, block_id, polish_tokens, POLISH_STEPS, POLISH_LR, teacher.device)
            if losses:
                print(f"     block {block_id}: polish {losses[0]:.4f} → {losses[-1]:.4f}")
            polished_weights[block_id] = _extract_polished_weights(single_student, block_id)
            del single_student

        # Post-polish PPL: inject polished weights into full stacked model
        _set_seed(seed)
        pack_full_post = _derive_block_pack_from_atlas(atlas, budget, FFN_BLOCKS)
        student_post = swap_compressed_ffns(teacher, pack_full_post, inplace=False)
        student_post.eval()
        for block_id in FFN_BLOCKS:
            if block_id in polished_weights and polished_weights[block_id]:
                _inject_polished_weights(student_post, block_id, polished_weights[block_id])

        post_metrics = _evaluate_model(teacher, student_post, corpus, batch_size=BATCH_SIZE)
        post_ppl = float(post_metrics["perplexity"])
        post_delta = (post_ppl - baseline_ppl) / baseline_ppl * 100.0
        print(f"     Post-polish PPL: {post_ppl:.3f} ({post_delta:+.2f}%)")

        del student_post

        seed_results[alloc_name] = {
            "pre_ppl": pre_ppl,
            "post_ppl": post_ppl,
            "pre_delta_pct": pre_delta,
            "post_delta_pct": post_delta,
            "total_compressed_params": total_compressed,
            "k_values": alloc,
        }

    per_seed[seed] = seed_results

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATING\n{'='*60}")


def _agg_post(alloc_name: str) -> tuple[float, float]:
    vals = [per_seed[s][alloc_name]["post_ppl"] for s in SEEDS]
    return float(np.mean(vals)), float(np.std(vals))


def _agg_pre(alloc_name: str) -> tuple[float, float]:
    vals = [per_seed[s][alloc_name]["pre_ppl"] for s in SEEDS]
    return float(np.mean(vals)), float(np.std(vals))


mean_baseline = float(np.mean([per_seed[s]["baseline"] for s in SEEDS]))
std_baseline = float(np.std([per_seed[s]["baseline"] for s in SEEDS]))

alloc_stats = {}
for alloc_name in ALLOCATIONS:
    mean_pre, std_pre = _agg_pre(alloc_name)
    mean_post, std_post = _agg_post(alloc_name)
    delta_pre = (mean_pre - mean_baseline) / mean_baseline * 100.0
    delta_post = (mean_post - mean_baseline) / mean_baseline * 100.0
    alloc_stats[alloc_name] = {
        "mean_pre": mean_pre, "std_pre": std_pre, "delta_pre_pct": delta_pre,
        "mean_post": mean_post, "std_post": std_post, "delta_post_pct": delta_post,
    }
    print(f"{alloc_name:15s}: pre={mean_pre:.3f}±{std_pre:.3f} ({delta_pre:+.2f}%)"
          f"  post={mean_post:.3f}±{std_post:.3f} ({delta_post:+.2f}%)")

# Verify budget matching across allocations
total_params = {}
for alloc_name in ALLOCATIONS:
    vals = [per_seed[s][alloc_name]["total_compressed_params"] for s in SEEDS]
    total_params[alloc_name] = vals[0]  # should be same across seeds

param_counts = list(total_params.values())
max_diff_pct = (max(param_counts) - min(param_counts)) / max(param_counts) * 100.0
print(f"\nParam count verification:")
for k, v in total_params.items():
    print(f"  {k}: {v:,} params")
print(f"  Max relative difference: {max_diff_pct:.3f}%")
assert max_diff_pct < 0.1, f"Budget mismatch exceeds 0.1%: {max_diff_pct:.3f}%"

# Verdict
uniform_post = alloc_stats["uniform"]["mean_post"]
best_nonunif_post = min(alloc_stats["middle_heavy"]["mean_post"], alloc_stats["edge_heavy"]["mean_post"])
delta_pct = (uniform_post - best_nonunif_post) / uniform_post * 100.0

# Positive delta = non-uniform is better (lower PPL)
if delta_pct >= 3.0:
    verdict = "CONFIRMED"
elif delta_pct < 1.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

best_alloc = "middle_heavy" if alloc_stats["middle_heavy"]["mean_post"] < alloc_stats["edge_heavy"]["mean_post"] else "edge_heavy"

print(f"\nVerdict: {verdict}")
print(f"  uniform post-polish PPL:     {uniform_post:.3f}")
print(f"  best non-uniform ({best_alloc}): {best_nonunif_post:.3f}")
print(f"  relative delta: {delta_pct:+.2f}%  (positive = non-uniform better)")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
results = {
    "cycle": 114,
    "item": 125,
    "seeds": SEEDS,
    "ffn_blocks": FFN_BLOCKS,
    "total_budget": TOTAL_BUDGET,
    "allocations": {k: dict(v) for k, v in ALLOCATIONS.items()},
    "polish_steps": POLISH_STEPS,
    "polish_lr": POLISH_LR,
    "baseline_ppl": {"mean": mean_baseline, "std": std_baseline},
    "allocation_stats": alloc_stats,
    "param_count_per_allocation": total_params,
    "param_budget_max_diff_pct": max_diff_pct,
    "uniform_post_ppl": uniform_post,
    "best_nonunif_post_ppl": best_nonunif_post,
    "best_nonunif_alloc": best_alloc,
    "relative_delta_pct": delta_pct,
    "verdict": verdict,
    "per_seed": {str(s): per_seed[s] for s in SEEDS},
}

json_path = RESULTS_DIR / "cycle_114_linearity_adaptive_ratio.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ---------------------------------------------------------------------------
# Plot: grouped bar chart (3 conditions × pre+post, with baseline reference)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

cond_labels = ["uniform", "middle_heavy", "edge_heavy"]
cond_display = ["Uniform\n(all k=512)", "Middle-heavy\n(mid k=384,\nedge k=640)", "Edge-heavy\n(edge k=384,\nmid k=640)"]
x = np.arange(len(cond_labels))
width = 0.35

pre_means = [alloc_stats[c]["mean_pre"] for c in cond_labels]
pre_stds = [alloc_stats[c]["std_pre"] for c in cond_labels]
post_means = [alloc_stats[c]["mean_post"] for c in cond_labels]
post_stds = [alloc_stats[c]["std_post"] for c in cond_labels]

bars_pre = ax.bar(x - width / 2, pre_means, width, label="Pre-polish", color="#90CAF9",
                  yerr=pre_stds, capsize=4, alpha=0.9)
bars_post = ax.bar(x + width / 2, post_means, width, label="Post-polish", color="#1565C0",
                   yerr=post_stds, capsize=4, alpha=0.9)

for i, (m, std) in enumerate(zip(post_means, post_stds)):
    d = (m - mean_baseline) / mean_baseline * 100.0
    ax.text(x[i] + width / 2, m + std + 0.3, f"{d:+.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.axhline(mean_baseline, color="green", linestyle="--", linewidth=1.5, alpha=0.8,
           label=f"Baseline {mean_baseline:.2f}")
ax.set_xticks(x)
ax.set_xticklabels(cond_display, fontsize=10)
ax.set_ylabel("Perplexity (OWT, 200 seqs)", fontsize=11)
ax.set_title(
    f"Linearity-Adaptive Ratio Allocation — GPT-2 FFN\n"
    f"Blocks {FFN_BLOCKS}, total k=2048 (matched budget)\n"
    f"Verdict: {verdict} | uniform={uniform_post:.2f}  best_nonunif({best_alloc})={best_nonunif_post:.2f}  Δ={delta_pct:+.2f}%",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.set_ylim(bottom=mean_baseline * 0.96)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

png_path = RESULTS_DIR / "cycle_114_linearity_adaptive_ratio.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved chart: {png_path}")

print(f"\n=== DONE ===")
print(f"Verdict: {verdict}")
for alloc_name in ALLOCATIONS:
    s = alloc_stats[alloc_name]
    print(f"  {alloc_name:15s}: pre={s['mean_pre']:.3f} ({s['delta_pre_pct']:+.2f}%)"
          f"  post={s['mean_post']:.3f} ({s['delta_post_pct']:+.2f}%)")
print(f"  uniform vs best non-uniform: {delta_pct:+.2f}% relative")
