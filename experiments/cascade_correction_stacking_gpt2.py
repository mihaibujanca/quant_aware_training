"""Cycle 115: Per-block cascade correction — GPT-2 FFN (item 118).

Compress k=4 FFN blocks [7,5,4,1] at 6x with topk+lstsq+500 Adam.
Freeze entire model. Insert learned corrections C_l = x + U_l @ (V_l @ x)
after each compressed block's full residual output. Train only {U_l, V_l}.

Tests ranks r in {4, 8, 16}, 3 seeds [42, 123, 7], 200 eval seqs OWT.

Verdict:
  CONFIRMED    if best reduction_pct >= 30% at r <= 16
  REFUTED      if reduction_pct < 10% even at r=16
  INCONCLUSIVE if 10-30% reduction
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

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

FFN_BLOCKS = [7, 5, 4, 1]
HIDDEN_FRACTION = 1 / 6
POLISH_STEPS = 500
POLISH_LR = 1e-4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200

CORRECTION_RANKS = [4, 8, 16]
CORRECTION_TRAIN_STEPS = 2000
CORRECTION_LR = 1e-4
D_MODEL = 768

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


def _param_count(w: torch.Tensor, b: Optional[torch.Tensor]) -> int:
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
# Low-rank correction module
# ---------------------------------------------------------------------------
class LowRankCorrection(nn.Module):
    """Additive low-rank perturbation of the residual stream.

    Forward: x_out = x + U @ (V @ x)
    U: (d, r), V: (r, d)
    Initialized with U = 0 so identity at start.
    """
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(d_model, rank))
        self.V = nn.Parameter(torch.randn(rank, d_model) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        return x + (x @ self.V.t()) @ self.U.t()


class CorrectedModel(nn.Module):
    """Wraps a compressed GPT-2 with per-block low-rank corrections.

    Corrections are applied via forward hooks on specified transformer blocks.
    """
    def __init__(self, base_model: nn.Module, block_ids: list[int], rank: int, d_model: int):
        super().__init__()
        self.base_model = base_model
        self.block_ids = sorted(block_ids)
        # Register corrections as submodules
        self.corrections = nn.ModuleDict({
            str(bid): LowRankCorrection(d_model, rank) for bid in block_ids
        })
        self._hooks: list = []

    def _install_hooks(self) -> None:
        self._remove_hooks()
        for bid in self.block_ids:
            correction = self.corrections[str(bid)]
            block = self.base_model.transformer.h[bid]

            def make_hook(corr):
                def hook(module, input, output):
                    # GPT-2 block output is a tuple: (hidden_states, present, ...)
                    if isinstance(output, tuple):
                        h = output[0]
                        h_corrected = corr(h)
                        return (h_corrected,) + output[1:]
                    else:
                        return corr(output)
                return hook

            h = block.register_forward_hook(make_hook(correction))
            self._hooks.append(h)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, input_ids: torch.Tensor):
        return self.base_model(input_ids)

    def __del__(self):
        self._remove_hooks()


def _train_corrections(
    corrected_model: CorrectedModel,
    train_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> list[float]:
    """Train only the correction parameters under next-token CE loss."""
    # Freeze everything
    corrected_model.base_model.requires_grad_(False)
    corrected_model.corrections.requires_grad_(True)

    params = list(corrected_model.corrections.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    corrected_model.base_model.eval()
    corrected_model.train()  # Put corrections in train mode

    losses = []
    n_seqs = train_tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(seed)

    for step in range(n_steps):
        idx = torch.randperm(n_seqs, generator=rng)[:BATCH_SIZE]
        batch = train_tokens[idx].to(device)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        optimizer.zero_grad()
        raw = corrected_model(inputs)
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if (step + 1) % 100 == 0:
            print(f"    step {step+1}/{n_steps}: loss={loss.item():.4f}")

    corrected_model.eval()
    return losses


def _eval_corrected(
    corrected_model: CorrectedModel,
    teacher: TeacherBundle,
    corpus,
) -> float:
    """Evaluate corrected model PPL by temporarily installing hooks."""
    corrected_model._install_hooks()
    corrected_model.eval()
    try:
        metrics = _evaluate_model(teacher, corrected_model.base_model, corpus, batch_size=BATCH_SIZE)
    finally:
        corrected_model._remove_hooks()
    return float(metrics["perplexity"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2 teacher...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()
print(f"  Teacher loaded on device={teacher.device}")

budget = CompressionBudgetSpec(
    hidden_fraction=HIDDEN_FRACTION,
    min_hidden=32,
    selection_mode="topk",
)

# ---------------------------------------------------------------------------
# Verify param counts
# ---------------------------------------------------------------------------
for r in CORRECTION_RANKS:
    expected = len(FFN_BLOCKS) * 2 * D_MODEL * r
    print(f"r={r}: expected correction params = {expected:,}")
    # r=4: 24576, r=8: 49152, r=16: 98304
    assert r == 4 and expected == 24576 or r == 8 and expected == 49152 or r == 16 and expected == 98304, \
        f"Unexpected param count {expected} for r={r}"
print("Param count check passed.")

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

    # Baseline float PPL
    bl_metrics = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = float(bl_metrics["perplexity"])
    print(f"  Baseline PPL: {baseline_ppl:.3f}")

    if not (25.0 <= baseline_ppl <= 65.0):
        print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [25, 65]")
        sys.exit(1)

    # Probe FFN geometry
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

    # Per-block FFN polish
    print(f"  Per-block polish ({POLISH_STEPS} steps)...")
    polished_weights: dict[int, dict] = {}
    for block_id in FFN_BLOCKS:
        _set_seed(seed)
        single_pack = _derive_block_pack_from_atlas(atlas, budget, [block_id])
        single_student = swap_compressed_ffns(teacher, single_pack, inplace=False)
        single_student.eval()
        losses = _polish_block(single_student, block_id, polish_tokens, POLISH_STEPS, POLISH_LR, teacher.device)
        if losses:
            print(f"    block {block_id}: polish {losses[0]:.4f} → {losses[-1]:.4f}")
        polished_weights[block_id] = _extract_polished_weights(single_student, block_id)
        del single_student

    # Build full compressed model (k=4, all polished)
    print("  Building full k=4 compressed model...")
    _set_seed(seed)
    pack = _derive_block_pack_from_atlas(atlas, budget, FFN_BLOCKS)
    compressed_model = swap_compressed_ffns(teacher, pack, inplace=False)
    compressed_model.eval()
    for bid in FFN_BLOCKS:
        if bid in polished_weights and polished_weights[bid]:
            _inject_polished_weights(compressed_model, bid, polished_weights[bid])

    # Eval uncorrected
    unc_metrics = _evaluate_model(teacher, compressed_model, corpus, batch_size=BATCH_SIZE)
    uncorrected_ppl = float(unc_metrics["perplexity"])
    degradation_base = uncorrected_ppl - baseline_ppl
    print(f"  Uncorrected PPL: {uncorrected_ppl:.3f}  (degradation={degradation_base:.3f})")

    if uncorrected_ppl > 2 * baseline_ppl:
        print(f"ABORT: uncorrected PPL {uncorrected_ppl:.2f} > 2x baseline")
        sys.exit(1)

    seed_results: dict = {
        "baseline": baseline_ppl,
        "uncorrected": uncorrected_ppl,
        "corrected": {},
        "loss_curves": {},
    }

    # Train corrections for each rank
    for rank in CORRECTION_RANKS:
        print(f"\n  Rank r={rank} correction training ({CORRECTION_TRAIN_STEPS} steps)...")
        _set_seed(seed)

        # Verify param count
        expected_params = len(FFN_BLOCKS) * 2 * D_MODEL * rank
        actual_correction = LowRankCorrection(D_MODEL, rank)
        actual_per_block = sum(p.numel() for p in actual_correction.parameters())
        total_actual = actual_per_block * len(FFN_BLOCKS)
        assert total_actual == expected_params, \
            f"Param count mismatch: {total_actual} != {expected_params}"
        print(f"    Correction params: {total_actual:,} (verified)")
        del actual_correction

        # Create a fresh copy of the compressed model for this rank/seed
        compressed_copy = copy.deepcopy(compressed_model)
        compressed_copy.eval()

        corrected = CorrectedModel(compressed_copy, FFN_BLOCKS, rank, D_MODEL)
        corrected._install_hooks()
        corrected = corrected.to(teacher.device)

        # Train
        train_tokens = corpus.probe_holdout  # Use holdout as training data for corrections
        loss_curve = _train_corrections(
            corrected,
            train_tokens,
            n_steps=CORRECTION_TRAIN_STEPS,
            lr=CORRECTION_LR,
            device=teacher.device,
            seed=seed,
        )
        print(f"    Train loss: {loss_curve[0]:.4f} → {loss_curve[-1]:.4f}")

        # Eval corrected model
        corrected_ppl = _eval_corrected(corrected, teacher, corpus)
        degradation_corrected = corrected_ppl - baseline_ppl
        reduction_pct = (degradation_base - degradation_corrected) / degradation_base * 100.0
        print(f"    r={rank}: corrected PPL={corrected_ppl:.3f}  reduction={reduction_pct:+.2f}%")

        seed_results["corrected"][rank] = corrected_ppl
        seed_results["loss_curves"][rank] = loss_curve

        del corrected, compressed_copy

    per_seed[seed] = seed_results
    del compressed_model

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATING\n{'='*60}")

mean_baseline = float(np.mean([per_seed[s]["baseline"] for s in SEEDS]))
std_baseline = float(np.std([per_seed[s]["baseline"] for s in SEEDS]))
mean_uncorrected = float(np.mean([per_seed[s]["uncorrected"] for s in SEEDS]))
std_uncorrected = float(np.std([per_seed[s]["uncorrected"] for s in SEEDS]))
degradation_base = mean_uncorrected - mean_baseline

print(f"Baseline:     PPL={mean_baseline:.3f} ± {std_baseline:.3f}")
print(f"Uncorrected:  PPL={mean_uncorrected:.3f} ± {std_uncorrected:.3f}  (deg={degradation_base:.3f})")

rank_results: dict[int, dict] = {}
best_reduction = -999.0
best_rank = -1

for rank in CORRECTION_RANKS:
    vals = [per_seed[s]["corrected"][rank] for s in SEEDS]
    mean_c = float(np.mean(vals))
    std_c = float(np.std(vals))
    deg_c = mean_c - mean_baseline
    reduction_pct = (degradation_base - deg_c) / degradation_base * 100.0
    rank_results[rank] = {
        "mean_ppl": mean_c,
        "std_ppl": std_c,
        "degradation": deg_c,
        "reduction_pct": reduction_pct,
    }
    print(f"r={rank}: PPL={mean_c:.3f} ± {std_c:.3f}  reduction={reduction_pct:+.2f}%")
    if reduction_pct > best_reduction:
        best_reduction = reduction_pct
        best_rank = rank

print(f"\nBest: r={best_rank} → {best_reduction:.2f}% reduction")

if best_reduction >= 30.0:
    verdict = "CONFIRMED"
elif best_reduction < 10.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
print(f"Verdict: {verdict}")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
results = {
    "cycle": 115,
    "item": 118,
    "seeds": SEEDS,
    "ffn_blocks": FFN_BLOCKS,
    "hidden_fraction": HIDDEN_FRACTION,
    "polish_steps": POLISH_STEPS,
    "polish_lr": POLISH_LR,
    "correction_ranks": CORRECTION_RANKS,
    "correction_train_steps": CORRECTION_TRAIN_STEPS,
    "correction_lr": CORRECTION_LR,
    "d_model": D_MODEL,
    "baseline_ppl": {"mean": mean_baseline, "std": std_baseline},
    "uncorrected_ppl": {"mean": mean_uncorrected, "std": std_uncorrected},
    "degradation_base": degradation_base,
    "rank_results": {str(r): rank_results[r] for r in CORRECTION_RANKS},
    "best_rank": best_rank,
    "best_reduction_pct": best_reduction,
    "verdict": verdict,
    "per_seed": {
        str(s): {
            "baseline": per_seed[s]["baseline"],
            "uncorrected": per_seed[s]["uncorrected"],
            "corrected": {str(r): per_seed[s]["corrected"][r] for r in CORRECTION_RANKS},
        }
        for s in SEEDS
    },
}

json_path = RESULTS_DIR / "cycle_115_cascade_correction.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# 1. Bar chart: PPL comparison
fig, ax = plt.subplots(figsize=(8, 5))
labels = ["float", "uncorrected"] + [f"r={r}" for r in CORRECTION_RANKS]
means = [mean_baseline, mean_uncorrected] + [rank_results[r]["mean_ppl"] for r in CORRECTION_RANKS]
stds = [std_baseline, std_uncorrected] + [rank_results[r]["std_ppl"] for r in CORRECTION_RANKS]
colors = ["#4CAF50", "#F44336"] + ["#2196F3", "#1976D2", "#0D47A1"]
x = np.arange(len(labels))
bars = ax.bar(x, means, yerr=stds, color=colors[:len(labels)], capsize=5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Perplexity")
ax.set_title(f"Cascade Correction PPL — GPT-2 k=4 6x\nVerdict: {verdict} (best r={best_rank}, {best_reduction:.1f}% reduction)")
ax.axhline(mean_baseline, color="#4CAF50", linestyle="--", alpha=0.5, label="float baseline")
# Annotate
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 0.3, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
ax.legend()
plt.tight_layout()
bar_path = RESULTS_DIR / "cycle_115_ppl_bar.png"
plt.savefig(bar_path, dpi=120)
plt.close()
print(f"Saved bar chart: {bar_path}")

# 2. Training loss curves per rank
fig, axes = plt.subplots(1, len(CORRECTION_RANKS), figsize=(14, 4), sharey=False)
for ax, rank in zip(axes, CORRECTION_RANKS):
    for s in SEEDS:
        curve = per_seed[s]["loss_curves"][rank]
        x_steps = list(range(1, len(curve) + 1))
        ax.plot(x_steps, curve, alpha=0.6, label=f"seed={s}")
    ax.set_title(f"r={rank}")
    ax.set_xlabel("Step")
    ax.set_ylabel("CE Loss")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
plt.suptitle("Correction Training Loss Curves")
plt.tight_layout()
curves_path = RESULTS_DIR / "cycle_115_loss_curves.png"
plt.savefig(curves_path, dpi=120)
plt.close()
print(f"Saved loss curves: {curves_path}")

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
manifest_path = RESULTS_DIR / "cycle_115_manifest.txt"
with open(manifest_path, "w") as f:
    f.write("Cycle 115: Per-block cascade correction — GPT-2 FFN\n")
    f.write("="*55 + "\n\n")
    f.write(f"Item: 118\n")
    f.write(f"Blocks compressed: {FFN_BLOCKS} at 6x (topk+lstsq+{POLISH_STEPS} Adam)\n")
    f.write(f"Correction: C_l(x) = x + U_l @ (V_l @ x), ranks {CORRECTION_RANKS}\n")
    f.write(f"Training: {CORRECTION_TRAIN_STEPS} Adam steps, lr={CORRECTION_LR}\n")
    f.write(f"Seeds: {SEEDS}, Eval seqs: {EVAL_SEQS}\n\n")
    f.write(f"Results:\n")
    f.write(f"  Baseline PPL:    {mean_baseline:.3f} ± {std_baseline:.3f}\n")
    f.write(f"  Uncorrected PPL: {mean_uncorrected:.3f} ± {std_uncorrected:.3f}\n")
    f.write(f"  Degradation:     {degradation_base:.3f}\n\n")
    for rank in CORRECTION_RANKS:
        rr = rank_results[rank]
        f.write(f"  r={rank}: PPL={rr['mean_ppl']:.3f} ± {rr['std_ppl']:.3f}  "
                f"reduction={rr['reduction_pct']:.1f}%  "
                f"params={len(FFN_BLOCKS)*2*D_MODEL*rank:,}\n")
    f.write(f"\nBest: r={best_rank} → {best_reduction:.1f}% reduction\n")
    f.write(f"Verdict: {verdict}\n\n")
    f.write(f"Files:\n")
    f.write(f"  {json_path}\n")
    f.write(f"  {bar_path}\n")
    f.write(f"  {curves_path}\n")
    f.write(f"  {manifest_path}\n")
    f.write(f"\nTSV row:\n")
    f.write(f"115\t118\tcompositional_stacking_cascade_correction_gpt2\t{verdict}\t"
            f"best_r={best_rank},reduction={best_reduction:.1f}%,"
            f"uncorrected_ppl={mean_uncorrected:.2f},corrected_ppl={rank_results[best_rank]['mean_ppl']:.2f},"
            f"baseline={mean_baseline:.2f}\n")

print(f"Saved manifest: {manifest_path}")
print("\nDone.")
