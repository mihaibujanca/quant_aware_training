"""Cycle 120: Cascade corrections at k=8 stacking — GPT-2 FFN (item 137).

Compress k=8 blocks [1,2,3,4,5,6,7,9] at 6x with topk+lstsq+500 Adam.
Freeze entire model. Insert learned corrections C_l = x + U_l @ (V_l @ x)
after each compressed block. Train only {U_l, V_l}.

r=8 primary (single seed 42). r=16 fallback if r=8 recovery < 10%.

Block set [1,2,3,4,5,6,7,9] = k=5 set {7,5,4,1,6} ∪ next 3 easiest {3,2,9}.
This is a proper superset of the k=5 set enabling valid scaling comparison.

Verdict:
  CONFIRMED    if recovery >= 20% (r <= 16)
  REFUTED      if recovery < 10% at r=16
  INCONCLUSIVE if 10-20% recovery, or r=16 needed but r=8 fails
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
SEEDS = [42]
EVAL_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4

FFN_BLOCKS = [1, 2, 3, 4, 5, 6, 7, 9]
CORRECTION_RANK_PRIMARY = 8
CORRECTION_RANK_FALLBACK = 16

HIDDEN_FRACTION = 1 / 6
POLISH_STEPS = 500
POLISH_LR = 1e-4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200

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
    corrected_model.base_model.requires_grad_(False)
    corrected_model.corrections.requires_grad_(True)

    params = list(corrected_model.corrections.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    corrected_model.base_model.eval()
    corrected_model.train()

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


def _compress_blocks(
    teacher: TeacherBundle,
    atlas: FfnGeometryAtlas,
    budget: CompressionBudgetSpec,
    block_ids: list[int],
    seed: int,
    corpus,
) -> nn.Module:
    """Build fully compressed+polished model for given block_ids."""
    polish_tokens = corpus.probe_holdout

    print(f"  Per-block polish ({POLISH_STEPS} steps) for {block_ids}...")
    polished_weights: dict[int, dict] = {}
    for block_id in block_ids:
        _set_seed(seed)
        single_pack = _derive_block_pack_from_atlas(atlas, budget, [block_id])
        single_student = swap_compressed_ffns(teacher, single_pack, inplace=False)
        single_student.eval()
        losses = _polish_block(single_student, block_id, polish_tokens, POLISH_STEPS, POLISH_LR, teacher.device)
        if losses:
            print(f"    block {block_id}: polish {losses[0]:.4f} -> {losses[-1]:.4f}")
        polished_weights[block_id] = _extract_polished_weights(single_student, block_id)
        del single_student

    print(f"  Building full compressed model for {block_ids}...")
    _set_seed(seed)
    pack = _derive_block_pack_from_atlas(atlas, budget, block_ids)
    compressed_model = swap_compressed_ffns(teacher, pack, inplace=False)
    compressed_model.eval()
    for bid in block_ids:
        if bid in polished_weights and polished_weights[bid]:
            _inject_polished_weights(compressed_model, bid, polished_weights[bid])

    return compressed_model


def _run_correction(
    compressed_model: nn.Module,
    teacher: TeacherBundle,
    corpus,
    block_ids: list[int],
    rank: int,
    correction_params_total: int,
    seed: int,
    uncorrected_ppl: float,
    baseline_ppl: float,
) -> dict:
    """Run correction training for a given rank. Returns result dict."""
    degradation = uncorrected_ppl - baseline_ppl
    print(f"\n  [k=8] Correction r={rank} ({CORRECTION_TRAIN_STEPS} steps)...")
    _set_seed(seed)
    compressed_copy = copy.deepcopy(compressed_model)
    corrected = CorrectedModel(compressed_copy, block_ids, rank, D_MODEL)
    corrected._install_hooks()
    corrected = corrected.to(teacher.device)

    loss_curve = _train_corrections(
        corrected, corpus.probe_holdout,
        n_steps=CORRECTION_TRAIN_STEPS,
        lr=CORRECTION_LR,
        device=teacher.device,
        seed=seed,
    )
    print(f"    Train loss: {loss_curve[0]:.4f} -> {loss_curve[-1]:.4f}")

    corrected_ppl = _eval_corrected(corrected, teacher, corpus)
    residual = corrected_ppl - baseline_ppl
    recovery = (degradation - residual) / degradation * 100.0
    print(f"  [k=8] Corrected PPL (r={rank}): {corrected_ppl:.3f}  recovery={recovery:.2f}%  params={correction_params_total:,}")

    del corrected, compressed_copy
    return {
        "corrected_ppl": corrected_ppl,
        "recovery": recovery,
        "loss_curve": loss_curve,
        "params": correction_params_total,
    }


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

correction_params_r8 = len(FFN_BLOCKS) * 2 * D_MODEL * CORRECTION_RANK_PRIMARY
correction_params_r16 = len(FFN_BLOCKS) * 2 * D_MODEL * CORRECTION_RANK_FALLBACK
print(f"k={len(FFN_BLOCKS)}, r={CORRECTION_RANK_PRIMARY}: correction params = {correction_params_r8:,}")
print(f"k={len(FFN_BLOCKS)}, r={CORRECTION_RANK_FALLBACK}: correction params = {correction_params_r16:,} (fallback)")

# Sanity check: P3 expected params
assert correction_params_r8 == 98304, f"Expected 98304, got {correction_params_r8}"
assert correction_params_r16 == 196608, f"Expected 196608, got {correction_params_r16}"
print("P3: Param counts verified.")

# ---------------------------------------------------------------------------
# Main loop: per-seed
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

    # P1: Sanity check
    if not (25.0 <= baseline_ppl <= 65.0):
        print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [25, 65]")
        sys.exit(1)

    # Probe and compress
    print(f"\n  [k=8] Probing FFN geometry for blocks {sorted(FFN_BLOCKS)}...")
    _set_seed(seed)
    atlas = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=tuple(sorted(FFN_BLOCKS)),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )

    compressed_model = _compress_blocks(teacher, atlas, budget, FFN_BLOCKS, seed, corpus)
    unc_metrics = _evaluate_model(teacher, compressed_model, corpus, batch_size=BATCH_SIZE)
    uncorrected_ppl = float(unc_metrics["perplexity"])
    degradation = uncorrected_ppl - baseline_ppl
    print(f"  [k=8] Uncorrected PPL: {uncorrected_ppl:.3f}  (degradation={degradation:.3f})")

    # P2: Sanity check
    if uncorrected_ppl <= baseline_ppl:
        print(f"WARNING: k=8 uncorrected PPL {uncorrected_ppl:.2f} <= baseline {baseline_ppl:.2f} — no degradation?")

    if uncorrected_ppl > 5000.0:
        print(f"ABORT seed {seed}: uncorrected PPL {uncorrected_ppl:.2f} > 5000, model collapsed, skipping.")
        per_seed[seed] = {
            "baseline": baseline_ppl,
            "uncorrected": uncorrected_ppl,
            "corrected_r8": None,
            "recovery_r8": None,
            "loss_curve_r8": [],
            "corrected_r16": None,
            "recovery_r16": None,
            "loss_curve_r16": [],
            "correction_params_r8": correction_params_r8,
            "correction_params_r16": correction_params_r16,
            "aborted": True,
        }
        del compressed_model, atlas, corpus
        continue
    elif uncorrected_ppl > 300.0:
        print(f"WARNING: uncorrected PPL {uncorrected_ppl:.2f} > 300 — severe degradation but proceeding with corrections.")

    # P5: Compare k=8 degradation to k=5 (cycle 119: ~70.4 PPL degradation ~70.1-51.7=18.4)
    # k=8 should be larger degradation than k=5's ~18.4 PPL
    k5_ref_deg = 70.1 - 51.7  # ~18.4
    if degradation < k5_ref_deg:
        print(f"WARNING: k=8 degradation {degradation:.2f} < k=5 reference {k5_ref_deg:.2f}. Unexpected.")

    # r=8 correction (primary)
    result_r8 = _run_correction(
        compressed_model, teacher, corpus, FFN_BLOCKS,
        rank=CORRECTION_RANK_PRIMARY,
        correction_params_total=correction_params_r8,
        seed=seed,
        uncorrected_ppl=uncorrected_ppl,
        baseline_ppl=baseline_ppl,
    )

    # Fallback r=16 if r=8 recovery < 10%
    result_r16 = None
    if result_r8["recovery"] < 10.0:
        print(f"\n  r=8 recovery {result_r8['recovery']:.2f}% < 10% threshold — running r=16 fallback...")
        result_r16 = _run_correction(
            compressed_model, teacher, corpus, FFN_BLOCKS,
            rank=CORRECTION_RANK_FALLBACK,
            correction_params_total=correction_params_r16,
            seed=seed,
            uncorrected_ppl=uncorrected_ppl,
            baseline_ppl=baseline_ppl,
        )
    else:
        print(f"\n  r=8 recovery {result_r8['recovery']:.2f}% >= 10% — skipping r=16 fallback.")

    per_seed[seed] = {
        "baseline": baseline_ppl,
        "uncorrected": uncorrected_ppl,
        "corrected_r8": result_r8["corrected_ppl"],
        "recovery_r8": result_r8["recovery"],
        "loss_curve_r8": result_r8["loss_curve"],
        "corrected_r16": result_r16["corrected_ppl"] if result_r16 else None,
        "recovery_r16": result_r16["recovery"] if result_r16 else None,
        "loss_curve_r16": result_r16["loss_curve"] if result_r16 else [],
        "correction_params_r8": correction_params_r8,
        "correction_params_r16": correction_params_r16,
        "aborted": False,
    }
    del compressed_model, atlas

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATING\n{'='*60}")

valid_seeds = [s for s in SEEDS if not per_seed[s].get("aborted", False) and per_seed[s]["corrected_r8"] is not None]
print(f"Valid seeds: {valid_seeds} (total {len(valid_seeds)}/{len(SEEDS)})")

if not valid_seeds:
    print("ERROR: no valid seeds! Cannot aggregate.")
    sys.exit(1)

mean_baseline = float(np.mean([per_seed[s]["baseline"] for s in valid_seeds]))
std_baseline = float(np.std([per_seed[s]["baseline"] for s in valid_seeds]))
mean_unc = float(np.mean([per_seed[s]["uncorrected"] for s in valid_seeds]))
std_unc = float(np.std([per_seed[s]["uncorrected"] for s in valid_seeds]))
deg = mean_unc - mean_baseline

vals_corr_r8 = [per_seed[s]["corrected_r8"] for s in valid_seeds]
mean_corr_r8 = float(np.mean(vals_corr_r8))
std_corr_r8 = float(np.std(vals_corr_r8))
recovery_r8 = (deg - (mean_corr_r8 - mean_baseline)) / deg * 100.0

r16_valid = [s for s in valid_seeds if per_seed[s]["corrected_r16"] is not None]
recovery_r16 = None
mean_corr_r16 = None
std_corr_r16 = None
if r16_valid:
    vals_corr_r16 = [per_seed[s]["corrected_r16"] for s in r16_valid]
    mean_corr_r16 = float(np.mean(vals_corr_r16))
    std_corr_r16 = float(np.std(vals_corr_r16))
    recovery_r16 = (deg - (mean_corr_r16 - mean_baseline)) / deg * 100.0

print(f"Baseline:        PPL={mean_baseline:.3f} ± {std_baseline:.3f}")
print(f"Uncorrected k=8: PPL={mean_unc:.3f} ± {std_unc:.3f}  deg={deg:.3f}")
print(f"Corrected  r=8:  PPL={mean_corr_r8:.3f} ± {std_corr_r8:.3f}  recovery={recovery_r8:.2f}%  params={correction_params_r8:,}")
if recovery_r16 is not None:
    print(f"Corrected  r=16: PPL={mean_corr_r16:.3f} ± {std_corr_r16:.3f}  recovery={recovery_r16:.2f}%  params={correction_params_r16:,}")

# Sanity checks
if not (49.7 <= mean_baseline <= 53.7):
    print(f"WARNING: baseline PPL {mean_baseline:.2f} outside ±2 of cycle 115 (51.7)")

# P4: Loss curve decreasing
for s in valid_seeds:
    lc = per_seed[s]["loss_curve_r8"]
    if lc and lc[0] <= lc[-1]:
        print(f"WARNING P4: seed {s} loss curve NOT decreasing ({lc[0]:.4f} -> {lc[-1]:.4f})")
    elif lc:
        print(f"P4: seed {s} loss decreasing: {lc[0]:.4f} -> {lc[-1]:.4f}  OK")

# P6: Recovery sanity
for rank_label, rec in [("r8", recovery_r8), ("r16", recovery_r16)]:
    if rec is not None and (rec > 100.0 or rec < -50.0):
        print(f"WARNING P6: recovery_{rank_label}={rec:.2f}% seems out of range")

# Verdict — use best available recovery
best_recovery = recovery_r8
best_rank = 8
if recovery_r16 is not None and recovery_r16 > recovery_r8:
    best_recovery = recovery_r16
    best_rank = 16

if best_recovery >= 20.0:
    verdict = "CONFIRMED"
elif recovery_r16 is not None and recovery_r16 < 10.0:
    verdict = "REFUTED"
elif recovery_r8 < 10.0 and recovery_r16 is None:
    # r=8 < 10% but r=16 not run (shouldn't happen due to fallback logic, but handle)
    verdict = "INCONCLUSIVE"
else:
    verdict = "INCONCLUSIVE"

print(f"\nVERDICT: {verdict}")
print(f"  k=8 recovery (r=8): {recovery_r8:.2f}%  (need >=20% for CONFIRMED)")
if recovery_r16 is not None:
    print(f"  k=8 recovery (r=16): {recovery_r16:.2f}%")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
results = {
    "cycle": 120,
    "item": 137,
    "seeds": SEEDS,
    "valid_seeds": valid_seeds,
    "ffn_blocks": FFN_BLOCKS,
    "hidden_fraction": HIDDEN_FRACTION,
    "polish_steps": POLISH_STEPS,
    "correction_rank_primary": CORRECTION_RANK_PRIMARY,
    "correction_rank_fallback": CORRECTION_RANK_FALLBACK,
    "correction_train_steps": CORRECTION_TRAIN_STEPS,
    "correction_lr": CORRECTION_LR,
    "d_model": D_MODEL,
    "baseline_ppl": {"mean": mean_baseline, "std": std_baseline},
    "uncorrected_ppl": {"mean": mean_unc, "std": std_unc},
    "degradation": deg,
    "corrected_r8_ppl": {"mean": mean_corr_r8, "std": std_corr_r8},
    "recovery_r8_pct": recovery_r8,
    "correction_params_r8": correction_params_r8,
    "corrected_r16_ppl": {"mean": mean_corr_r16, "std": std_corr_r16} if mean_corr_r16 is not None else None,
    "recovery_r16_pct": recovery_r16,
    "correction_params_r16": correction_params_r16 if r16_valid else None,
    "verdict": verdict,
    "best_recovery_pct": best_recovery,
    "best_rank": best_rank,
    "per_seed": {
        str(s): {
            "baseline": per_seed[s]["baseline"],
            "uncorrected": per_seed[s]["uncorrected"],
            "corrected_r8": per_seed[s]["corrected_r8"],
            "recovery_r8": per_seed[s]["recovery_r8"],
            "corrected_r16": per_seed[s]["corrected_r16"],
            "recovery_r16": per_seed[s]["recovery_r16"],
            "aborted": per_seed[s].get("aborted", False),
        }
        for s in SEEDS
    },
    # Reference points for comparison
    "reference": {
        "k4_cycle115": {"baseline": 51.7, "uncorrected": 80.7, "corrected": 59.1, "recovery": 74.3, "rank": 8},
        "k5_cycle119": {"baseline": 51.7, "uncorrected": 122.1, "corrected": 70.1, "recovery": 73.8, "rank": 8},
    },
}

json_path = RESULTS_DIR / "cycle_120_cascade_correction_k8.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ---------------------------------------------------------------------------
# Plot — bar chart + loss curves
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
ax = axes[0]
labels_bar = ["float", "unc k=8", f"corr k=8\nr=8"]
means_vals = [mean_baseline, mean_unc, mean_corr_r8]
stds_vals = [std_baseline, std_unc, std_corr_r8]
colors_bar = ["#4CAF50", "#F44336", "#2196F3"]
if mean_corr_r16 is not None:
    labels_bar.append(f"corr k=8\nr=16")
    means_vals.append(mean_corr_r16)
    stds_vals.append(std_corr_r16 if std_corr_r16 else 0.0)
    colors_bar.append("#9C27B0")

x = np.arange(len(labels_bar))
bars = ax.bar(x, means_vals, yerr=stds_vals, color=colors_bar, capsize=5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels_bar)
ax.set_ylabel("Perplexity")
ax.set_title(f"Cascade Correction — GPT-2 k=8 at 6x\nVerdict: {verdict} | recovery={best_recovery:.1f}% (r={best_rank})")
ax.axhline(mean_baseline, color="#4CAF50", linestyle="--", alpha=0.5, label="float baseline")

# Reference lines
k4_ref = {"baseline": 51.7, "uncorrected": 80.7, "corrected": 59.1, "recovery": 74.3}
k5_ref = {"baseline": 51.7, "uncorrected": 122.1, "corrected": 70.1, "recovery": 73.8}
ax.axhline(k4_ref["corrected"], color="#FF9800", linestyle=":", alpha=0.6, label=f"k=4 corr (cyc115) {k4_ref['corrected']:.1f}")
ax.axhline(k5_ref["corrected"], color="#795548", linestyle=":", alpha=0.6, label=f"k=5 corr (cyc119) {k5_ref['corrected']:.1f}")

for bar, m in zip(bars, means_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 0.5, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
ax.legend(fontsize=7)

# Loss curves
ax2 = axes[1]
for s in valid_seeds:
    curve_r8 = per_seed[s]["loss_curve_r8"]
    ax2.plot(range(1, len(curve_r8) + 1), curve_r8, alpha=0.7, label=f"seed={s} r=8", linestyle="-")
    curve_r16 = per_seed[s].get("loss_curve_r16", [])
    if curve_r16:
        ax2.plot(range(1, len(curve_r16) + 1), curve_r16, alpha=0.7, label=f"seed={s} r=16", linestyle="--")

ax2.set_title("Correction Training Loss (k=8)")
ax2.set_xlabel("Step")
ax2.set_ylabel("CE Loss")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "cycle_120_cascade_correction_k8.png"
plt.savefig(plot_path, dpi=120)
plt.close()
print(f"Saved plot: {plot_path}")

print("\nDone.")
