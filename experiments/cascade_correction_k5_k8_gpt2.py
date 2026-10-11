"""Cycle 117: Cascade corrections at k=5 and k=8 — GPT-2 FFN (item 133).

Compress k=5 blocks [7,5,4,1,6] and k=8 blocks [7,5,4,3,2,1,0,6] at 6x
with topk+lstsq+500 Adam. Freeze entire model. Insert learned corrections
C_l = x + U_l @ (V_l @ x) after each compressed block. Train only {U_l, V_l}.

Primary rank r=8. Fallback r=16 at k=8 if r=8 recovery < 20%.
3 seeds [42, 123, 7], 200 eval seqs OWT.

Verdict:
  CONFIRMED    if >=30% recovery at k=5 AND >=20% at k=8 with r<=16
  REFUTED      if <10% recovery at k=5
  INCONCLUSIVE if works at k=5 but fails at k=8, or needs r>16
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

FFN_BLOCKS_K5 = [7, 5, 4, 1, 6]
FFN_BLOCKS_K8 = [7, 5, 4, 3, 2, 1, 0, 6]

HIDDEN_FRACTION = 1 / 6
POLISH_STEPS = 500
POLISH_LR = 1e-4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200

CORRECTION_RANK_PRIMARY = 8
CORRECTION_RANK_FALLBACK = 16
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
for k_blocks, r in [(FFN_BLOCKS_K5, CORRECTION_RANK_PRIMARY), (FFN_BLOCKS_K8, CORRECTION_RANK_PRIMARY),
                    (FFN_BLOCKS_K8, CORRECTION_RANK_FALLBACK)]:
    expected = len(k_blocks) * 2 * D_MODEL * r
    print(f"k={len(k_blocks)}, r={r}: correction params = {expected:,}")
print("Param count check complete.")

# ---------------------------------------------------------------------------
# Main loop: per-seed, per-k
# ---------------------------------------------------------------------------
per_seed_k5: dict[int, dict] = {}
per_seed_k8: dict[int, dict] = {}

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

    # --- k=5 ---
    print(f"\n  [k=5] Probing FFN geometry for blocks {sorted(FFN_BLOCKS_K5)}...")
    _set_seed(seed)
    atlas_k5 = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=tuple(sorted(FFN_BLOCKS_K5)),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )

    compressed_k5 = _compress_blocks(teacher, atlas_k5, budget, FFN_BLOCKS_K5, seed, corpus)
    unc_metrics_k5 = _evaluate_model(teacher, compressed_k5, corpus, batch_size=BATCH_SIZE)
    uncorrected_ppl_k5 = float(unc_metrics_k5["perplexity"])
    degradation_k5 = uncorrected_ppl_k5 - baseline_ppl
    print(f"  [k=5] Uncorrected PPL: {uncorrected_ppl_k5:.3f}  (degradation={degradation_k5:.3f})")

    if not (55.0 <= uncorrected_ppl_k5 <= 120.0):
        print(f"WARNING: k=5 uncorrected PPL {uncorrected_ppl_k5:.2f} outside [55, 120]; proceeding anyway")

    # k=5, r=8 correction
    print(f"\n  [k=5] Correction r={CORRECTION_RANK_PRIMARY} ({CORRECTION_TRAIN_STEPS} steps)...")
    correction_params_k5 = len(FFN_BLOCKS_K5) * 2 * D_MODEL * CORRECTION_RANK_PRIMARY
    _set_seed(seed)
    compressed_copy_k5 = copy.deepcopy(compressed_k5)
    corrected_k5 = CorrectedModel(compressed_copy_k5, FFN_BLOCKS_K5, CORRECTION_RANK_PRIMARY, D_MODEL)
    corrected_k5._install_hooks()
    corrected_k5 = corrected_k5.to(teacher.device)

    loss_curve_k5 = _train_corrections(
        corrected_k5, corpus.probe_holdout,
        n_steps=CORRECTION_TRAIN_STEPS,
        lr=CORRECTION_LR,
        device=teacher.device,
        seed=seed,
    )
    print(f"    Train loss: {loss_curve_k5[0]:.4f} -> {loss_curve_k5[-1]:.4f}")

    corrected_ppl_k5 = _eval_corrected(corrected_k5, teacher, corpus)
    residual_k5 = corrected_ppl_k5 - baseline_ppl
    recovery_k5 = (degradation_k5 - residual_k5) / degradation_k5 * 100.0
    print(f"  [k=5] Corrected PPL: {corrected_ppl_k5:.3f}  recovery={recovery_k5:.2f}%  params={correction_params_k5:,}")

    per_seed_k5[seed] = {
        "baseline": baseline_ppl,
        "uncorrected": uncorrected_ppl_k5,
        "corrected_r8": corrected_ppl_k5,
        "recovery_r8": recovery_k5,
        "loss_curve_r8": loss_curve_k5,
        "correction_params": correction_params_k5,
    }
    del corrected_k5, compressed_copy_k5, compressed_k5, atlas_k5

    # --- k=8 ---
    print(f"\n  [k=8] Probing FFN geometry for blocks {sorted(FFN_BLOCKS_K8)}...")
    _set_seed(seed)
    atlas_k8 = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=tuple(sorted(FFN_BLOCKS_K8)),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )

    compressed_k8 = _compress_blocks(teacher, atlas_k8, budget, FFN_BLOCKS_K8, seed, corpus)
    unc_metrics_k8 = _evaluate_model(teacher, compressed_k8, corpus, batch_size=BATCH_SIZE)
    uncorrected_ppl_k8 = float(unc_metrics_k8["perplexity"])
    degradation_k8 = uncorrected_ppl_k8 - baseline_ppl
    print(f"  [k=8] Uncorrected PPL: {uncorrected_ppl_k8:.3f}  (degradation={degradation_k8:.3f})")

    if not (100.0 <= uncorrected_ppl_k8 <= 1000.0):
        print(f"WARNING: k=8 uncorrected PPL {uncorrected_ppl_k8:.2f} outside [100, 1000]; proceeding anyway")

    # k=8, r=8 correction
    print(f"\n  [k=8] Correction r={CORRECTION_RANK_PRIMARY} ({CORRECTION_TRAIN_STEPS} steps)...")
    correction_params_k8 = len(FFN_BLOCKS_K8) * 2 * D_MODEL * CORRECTION_RANK_PRIMARY
    _set_seed(seed)
    compressed_copy_k8 = copy.deepcopy(compressed_k8)
    corrected_k8 = CorrectedModel(compressed_copy_k8, FFN_BLOCKS_K8, CORRECTION_RANK_PRIMARY, D_MODEL)
    corrected_k8._install_hooks()
    corrected_k8 = corrected_k8.to(teacher.device)

    loss_curve_k8 = _train_corrections(
        corrected_k8, corpus.probe_holdout,
        n_steps=CORRECTION_TRAIN_STEPS,
        lr=CORRECTION_LR,
        device=teacher.device,
        seed=seed,
    )
    print(f"    Train loss: {loss_curve_k8[0]:.4f} -> {loss_curve_k8[-1]:.4f}")

    corrected_ppl_k8_r8 = _eval_corrected(corrected_k8, teacher, corpus)
    residual_k8_r8 = corrected_ppl_k8_r8 - baseline_ppl
    recovery_k8_r8 = (degradation_k8 - residual_k8_r8) / degradation_k8 * 100.0
    print(f"  [k=8] r=8 Corrected PPL: {corrected_ppl_k8_r8:.3f}  recovery={recovery_k8_r8:.2f}%  params={correction_params_k8:,}")

    per_seed_k8[seed] = {
        "baseline": baseline_ppl,
        "uncorrected": uncorrected_ppl_k8,
        "corrected_r8": corrected_ppl_k8_r8,
        "recovery_r8": recovery_k8_r8,
        "loss_curve_r8": loss_curve_k8,
        "correction_params_r8": correction_params_k8,
        "corrected_r16": None,
        "recovery_r16": None,
        "loss_curve_r16": None,
        "correction_params_r16": None,
    }
    del corrected_k8, compressed_copy_k8

    # Still hold compressed_k8 for potential fallback
    per_seed_k8[seed]["_compressed_k8_for_fallback"] = compressed_k8
    per_seed_k8[seed]["_atlas_k8_for_fallback"] = atlas_k8

# ---------------------------------------------------------------------------
# Check if fallback is needed for k=8
# ---------------------------------------------------------------------------
mean_recovery_k8_r8 = float(np.mean([per_seed_k8[s]["recovery_r8"] for s in SEEDS]))
fallback_triggered = mean_recovery_k8_r8 < 20.0
print(f"\nMean k=8 r=8 recovery: {mean_recovery_k8_r8:.2f}%")

if fallback_triggered:
    print(f"FALLBACK TRIGGERED: mean recovery {mean_recovery_k8_r8:.2f}% < 20% — running r=16 at k=8")
    for seed in SEEDS:
        print(f"\n  [k=8 r=16 fallback] SEED {seed}")
        _set_seed(seed)

        # Rebuild corpus for consistent eval
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

        compressed_k8 = per_seed_k8[seed].pop("_compressed_k8_for_fallback")
        correction_params_k8_r16 = len(FFN_BLOCKS_K8) * 2 * D_MODEL * CORRECTION_RANK_FALLBACK

        _set_seed(seed)
        compressed_copy_k8_r16 = copy.deepcopy(compressed_k8)
        corrected_k8_r16 = CorrectedModel(compressed_copy_k8_r16, FFN_BLOCKS_K8, CORRECTION_RANK_FALLBACK, D_MODEL)
        corrected_k8_r16._install_hooks()
        corrected_k8_r16 = corrected_k8_r16.to(teacher.device)

        loss_curve_k8_r16 = _train_corrections(
            corrected_k8_r16, corpus.probe_holdout,
            n_steps=CORRECTION_TRAIN_STEPS,
            lr=CORRECTION_LR,
            device=teacher.device,
            seed=seed,
        )
        print(f"    Train loss: {loss_curve_k8_r16[0]:.4f} -> {loss_curve_k8_r16[-1]:.4f}")

        corrected_ppl_k8_r16 = _eval_corrected(corrected_k8_r16, teacher, corpus)
        degradation_k8 = per_seed_k8[seed]["uncorrected"] - per_seed_k8[seed]["baseline"]
        residual_k8_r16 = corrected_ppl_k8_r16 - per_seed_k8[seed]["baseline"]
        recovery_k8_r16 = (degradation_k8 - residual_k8_r16) / degradation_k8 * 100.0
        print(f"  [k=8] r=16 Corrected PPL: {corrected_ppl_k8_r16:.3f}  recovery={recovery_k8_r16:.2f}%  params={correction_params_k8_r16:,}")

        per_seed_k8[seed]["corrected_r16"] = corrected_ppl_k8_r16
        per_seed_k8[seed]["recovery_r16"] = recovery_k8_r16
        per_seed_k8[seed]["loss_curve_r16"] = loss_curve_k8_r16
        per_seed_k8[seed]["correction_params_r16"] = correction_params_k8_r16
        del corrected_k8_r16, compressed_copy_k8_r16, compressed_k8
        per_seed_k8[seed].pop("_atlas_k8_for_fallback", None)
else:
    # Cleanup fallback references
    for seed in SEEDS:
        per_seed_k8[seed].pop("_compressed_k8_for_fallback", None)
        per_seed_k8[seed].pop("_atlas_k8_for_fallback", None)

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATING\n{'='*60}")

mean_baseline = float(np.mean([per_seed_k5[s]["baseline"] for s in SEEDS]))
std_baseline = float(np.std([per_seed_k5[s]["baseline"] for s in SEEDS]))

# k=5
mean_unc_k5 = float(np.mean([per_seed_k5[s]["uncorrected"] for s in SEEDS]))
std_unc_k5 = float(np.std([per_seed_k5[s]["uncorrected"] for s in SEEDS]))
deg_k5 = mean_unc_k5 - mean_baseline
vals_corr_k5_r8 = [per_seed_k5[s]["corrected_r8"] for s in SEEDS]
mean_corr_k5_r8 = float(np.mean(vals_corr_k5_r8))
std_corr_k5_r8 = float(np.std(vals_corr_k5_r8))
recovery_k5_r8 = (deg_k5 - (mean_corr_k5_r8 - mean_baseline)) / deg_k5 * 100.0
correction_params_k5_final = len(FFN_BLOCKS_K5) * 2 * D_MODEL * CORRECTION_RANK_PRIMARY

# k=8
mean_unc_k8 = float(np.mean([per_seed_k8[s]["uncorrected"] for s in SEEDS]))
std_unc_k8 = float(np.std([per_seed_k8[s]["uncorrected"] for s in SEEDS]))
deg_k8 = mean_unc_k8 - mean_baseline
vals_corr_k8_r8 = [per_seed_k8[s]["corrected_r8"] for s in SEEDS]
mean_corr_k8_r8 = float(np.mean(vals_corr_k8_r8))
std_corr_k8_r8 = float(np.std(vals_corr_k8_r8))
recovery_k8_r8_final = (deg_k8 - (mean_corr_k8_r8 - mean_baseline)) / deg_k8 * 100.0
correction_params_k8_final = len(FFN_BLOCKS_K8) * 2 * D_MODEL * CORRECTION_RANK_PRIMARY

# r=16 fallback stats if triggered
mean_corr_k8_r16 = None
std_corr_k8_r16 = None
recovery_k8_r16_final = None
correction_params_k8_r16_final = None
if fallback_triggered:
    vals_corr_k8_r16 = [per_seed_k8[s]["corrected_r16"] for s in SEEDS]
    mean_corr_k8_r16 = float(np.mean(vals_corr_k8_r16))
    std_corr_k8_r16 = float(np.std(vals_corr_k8_r16))
    recovery_k8_r16_final = (deg_k8 - (mean_corr_k8_r16 - mean_baseline)) / deg_k8 * 100.0
    correction_params_k8_r16_final = len(FFN_BLOCKS_K8) * 2 * D_MODEL * CORRECTION_RANK_FALLBACK

print(f"Baseline:           PPL={mean_baseline:.3f} ± {std_baseline:.3f}")
print(f"[k=5] Uncorrected:  PPL={mean_unc_k5:.3f} ± {std_unc_k5:.3f}  deg={deg_k5:.3f}")
print(f"[k=5] Corrected r=8: PPL={mean_corr_k5_r8:.3f} ± {std_corr_k5_r8:.3f}  recovery={recovery_k5_r8:.2f}%  params={correction_params_k5_final:,}")
print(f"[k=8] Uncorrected:  PPL={mean_unc_k8:.3f} ± {std_unc_k8:.3f}  deg={deg_k8:.3f}")
print(f"[k=8] Corrected r=8: PPL={mean_corr_k8_r8:.3f} ± {std_corr_k8_r8:.3f}  recovery={recovery_k8_r8_final:.2f}%  params={correction_params_k8_final:,}")
if fallback_triggered:
    print(f"[k=8] Corrected r=16 (fallback): PPL={mean_corr_k8_r16:.3f} ± {std_corr_k8_r16:.3f}  recovery={recovery_k8_r16_final:.2f}%  params={correction_params_k8_r16_final:,}")

# Best k=8 recovery (r=8 or fallback r=16)
best_recovery_k8 = recovery_k8_r8_final
best_rank_k8 = CORRECTION_RANK_PRIMARY
best_ppl_k8 = mean_corr_k8_r8
if fallback_triggered and recovery_k8_r16_final is not None and recovery_k8_r16_final > recovery_k8_r8_final:
    best_recovery_k8 = recovery_k8_r16_final
    best_rank_k8 = CORRECTION_RANK_FALLBACK
    best_ppl_k8 = mean_corr_k8_r16

# Verdict
if recovery_k5_r8 >= 30.0 and best_recovery_k8 >= 20.0:
    verdict = "CONFIRMED"
elif recovery_k5_r8 < 10.0:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\nVERDICT: {verdict}")
print(f"  k=5 recovery (r=8): {recovery_k5_r8:.2f}%  (need >=30%)")
print(f"  k=8 best recovery (r={best_rank_k8}): {best_recovery_k8:.2f}%  (need >=20%)")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
results = {
    "cycle": 117,
    "item": 133,
    "seeds": SEEDS,
    "ffn_blocks_k5": FFN_BLOCKS_K5,
    "ffn_blocks_k8": FFN_BLOCKS_K8,
    "hidden_fraction": HIDDEN_FRACTION,
    "polish_steps": POLISH_STEPS,
    "correction_rank_primary": CORRECTION_RANK_PRIMARY,
    "correction_rank_fallback": CORRECTION_RANK_FALLBACK,
    "fallback_triggered": fallback_triggered,
    "correction_train_steps": CORRECTION_TRAIN_STEPS,
    "correction_lr": CORRECTION_LR,
    "d_model": D_MODEL,
    "baseline_ppl": {"mean": mean_baseline, "std": std_baseline},
    "k5": {
        "uncorrected_ppl": {"mean": mean_unc_k5, "std": std_unc_k5},
        "degradation": deg_k5,
        "corrected_r8_ppl": {"mean": mean_corr_k5_r8, "std": std_corr_k5_r8},
        "recovery_r8_pct": recovery_k5_r8,
        "correction_params_r8": correction_params_k5_final,
    },
    "k8": {
        "uncorrected_ppl": {"mean": mean_unc_k8, "std": std_unc_k8},
        "degradation": deg_k8,
        "corrected_r8_ppl": {"mean": mean_corr_k8_r8, "std": std_corr_k8_r8},
        "recovery_r8_pct": recovery_k8_r8_final,
        "correction_params_r8": correction_params_k8_final,
        "corrected_r16_ppl": {"mean": mean_corr_k8_r16, "std": std_corr_k8_r16} if fallback_triggered else None,
        "recovery_r16_pct": recovery_k8_r16_final,
        "correction_params_r16": correction_params_k8_r16_final,
        "best_recovery_pct": best_recovery_k8,
        "best_rank": best_rank_k8,
    },
    "verdict": verdict,
    "per_seed_k5": {
        str(s): {
            "baseline": per_seed_k5[s]["baseline"],
            "uncorrected": per_seed_k5[s]["uncorrected"],
            "corrected_r8": per_seed_k5[s]["corrected_r8"],
            "recovery_r8": per_seed_k5[s]["recovery_r8"],
        }
        for s in SEEDS
    },
    "per_seed_k8": {
        str(s): {
            "baseline": per_seed_k8[s]["baseline"],
            "uncorrected": per_seed_k8[s]["uncorrected"],
            "corrected_r8": per_seed_k8[s]["corrected_r8"],
            "recovery_r8": per_seed_k8[s]["recovery_r8"],
            "corrected_r16": per_seed_k8[s].get("corrected_r16"),
            "recovery_r16": per_seed_k8[s].get("recovery_r16"),
        }
        for s in SEEDS
    },
}

json_path = RESULTS_DIR / "cycle_117_cascade_k5_k8.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# Grouped bar chart: float | unc_k5 | corr_k5_r8 | unc_k8 | corr_k8_best
fig, ax = plt.subplots(figsize=(10, 5))

labels = ["float", "unc k=5", f"corr k=5\nr=8", "unc k=8", f"corr k=8\nr={best_rank_k8}"]
means_vals = [mean_baseline, mean_unc_k5, mean_corr_k5_r8, mean_unc_k8, best_ppl_k8]
stds_vals = [std_baseline, std_unc_k5, std_corr_k5_r8, std_unc_k8,
             std_corr_k8_r8 if best_rank_k8 == CORRECTION_RANK_PRIMARY else (std_corr_k8_r16 or std_corr_k8_r8)]
colors_bar = ["#4CAF50", "#F44336", "#2196F3", "#D32F2F", "#1565C0"]

x = np.arange(len(labels))
bars = ax.bar(x, means_vals, yerr=stds_vals, color=colors_bar, capsize=5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Perplexity")
ax.set_title(f"Cascade Correction — GPT-2 k=5 and k=8 at 6x\n"
             f"Verdict: {verdict} | k5 recovery={recovery_k5_r8:.1f}% | k8 recovery={best_recovery_k8:.1f}%")
ax.axhline(mean_baseline, color="#4CAF50", linestyle="--", alpha=0.5, label="float baseline")
for bar, m in zip(bars, means_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 0.5, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
ax.legend()
plt.tight_layout()
bar_path = RESULTS_DIR / "cycle_117_ppl_bar.png"
plt.savefig(bar_path, dpi=120)
plt.close()
print(f"Saved bar chart: {bar_path}")

# Loss curves: 2x2 grid (k5/k8 x r8/fallback if applicable)
n_plots = 4 if fallback_triggered else 2
fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=False)
if n_plots == 1:
    axes = [axes]

for s in SEEDS:
    curve = per_seed_k5[s]["loss_curve_r8"]
    axes[0].plot(range(1, len(curve) + 1), curve, alpha=0.6, label=f"seed={s}")
axes[0].set_title("k=5, r=8")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("CE Loss")
axes[0].legend(fontsize=7)
axes[0].grid(alpha=0.3)

for s in SEEDS:
    curve = per_seed_k8[s]["loss_curve_r8"]
    axes[1].plot(range(1, len(curve) + 1), curve, alpha=0.6, label=f"seed={s}")
axes[1].set_title("k=8, r=8")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("CE Loss")
axes[1].legend(fontsize=7)
axes[1].grid(alpha=0.3)

if fallback_triggered and n_plots >= 4:
    for s in SEEDS:
        curve = per_seed_k8[s].get("loss_curve_r16") or []
        if curve:
            axes[2].plot(range(1, len(curve) + 1), curve, alpha=0.6, label=f"seed={s}")
    axes[2].set_title("k=8, r=16 (fallback)")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("CE Loss")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    # Recovery comparison plot
    k_labels = ["k=5 r=8", "k=8 r=8", "k=8 r=16"]
    recoveries = [recovery_k5_r8, recovery_k8_r8_final, recovery_k8_r16_final or 0]
    axes[3].bar(k_labels, recoveries, color=["#2196F3", "#1976D2", "#0D47A1"])
    axes[3].axhline(30, color="orange", linestyle="--", alpha=0.7, label="30% threshold")
    axes[3].axhline(20, color="red", linestyle="--", alpha=0.7, label="20% threshold")
    axes[3].set_ylabel("Recovery %")
    axes[3].set_title("Recovery Summary")
    axes[3].legend(fontsize=7)

plt.suptitle("Correction Training Loss Curves")
plt.tight_layout()
curves_path = RESULTS_DIR / "cycle_117_loss_curves.png"
plt.savefig(curves_path, dpi=120)
plt.close()
print(f"Saved loss curves: {curves_path}")

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
manifest_path = RESULTS_DIR / "cycle_117_manifest.txt"
with open(manifest_path, "w") as f:
    f.write("Cycle 117: Cascade corrections at k=5 and k=8 — GPT-2 FFN\n")
    f.write("=" * 58 + "\n\n")
    f.write(f"Item: 133\n")
    f.write(f"k=5 blocks: {FFN_BLOCKS_K5}\n")
    f.write(f"k=8 blocks: {FFN_BLOCKS_K8}\n")
    f.write(f"Compression: 6x (hidden_fraction=1/6), topk+lstsq+{POLISH_STEPS} Adam\n")
    f.write(f"Corrections: C_l(x) = x + U_l @ (V_l @ x), r=8 primary, r=16 fallback\n")
    f.write(f"Correction training: {CORRECTION_TRAIN_STEPS} Adam steps, lr={CORRECTION_LR}\n")
    f.write(f"Seeds: {SEEDS}, Eval seqs: {EVAL_SEQS}\n\n")
    f.write(f"Results:\n")
    f.write(f"  Baseline PPL:             {mean_baseline:.3f} ± {std_baseline:.3f}\n\n")
    f.write(f"  [k=5]\n")
    f.write(f"    Uncorrected PPL:        {mean_unc_k5:.3f} ± {std_unc_k5:.3f}  (deg={deg_k5:.3f})\n")
    f.write(f"    Corrected r=8 PPL:      {mean_corr_k5_r8:.3f} ± {std_corr_k5_r8:.3f}\n")
    f.write(f"    Recovery r=8:           {recovery_k5_r8:.1f}%\n")
    f.write(f"    Correction params r=8:  {correction_params_k5_final:,}\n\n")
    f.write(f"  [k=8]\n")
    f.write(f"    Uncorrected PPL:        {mean_unc_k8:.3f} ± {std_unc_k8:.3f}  (deg={deg_k8:.3f})\n")
    f.write(f"    Corrected r=8 PPL:      {mean_corr_k8_r8:.3f} ± {std_corr_k8_r8:.3f}\n")
    f.write(f"    Recovery r=8:           {recovery_k8_r8_final:.1f}%\n")
    f.write(f"    Correction params r=8:  {correction_params_k8_final:,}\n")
    if fallback_triggered:
        f.write(f"    Fallback triggered (r=8 recovery={recovery_k8_r8_final:.1f}% < 20%)\n")
        f.write(f"    Corrected r=16 PPL:     {mean_corr_k8_r16:.3f} ± {std_corr_k8_r16:.3f}\n")
        f.write(f"    Recovery r=16:          {recovery_k8_r16_final:.1f}%\n")
        f.write(f"    Correction params r=16: {correction_params_k8_r16_final:,}\n")
    f.write(f"\n  Best k=8 recovery: r={best_rank_k8}, {best_recovery_k8:.1f}%\n")
    f.write(f"\nVerdict: {verdict}\n")
    f.write(f"  k=5 recovery={recovery_k5_r8:.1f}% (need >=30%)\n")
    f.write(f"  k=8 recovery={best_recovery_k8:.1f}% (need >=20%, r<={CORRECTION_RANK_FALLBACK})\n\n")
    f.write(f"Files:\n")
    f.write(f"  {json_path}\n")
    f.write(f"  {bar_path}\n")
    f.write(f"  {curves_path}\n")
    f.write(f"  {manifest_path}\n")
    f.write(f"\nTSV row:\n")
    tsv_notes = (
        f"baseline_ppl={mean_baseline:.2f} "
        f"unc_k5={mean_unc_k5:.2f} corr_k5_r8={mean_corr_k5_r8:.2f} recovery_k5={recovery_k5_r8:.1f}% "
        f"unc_k8={mean_unc_k8:.2f} corr_k8_r{best_rank_k8}={best_ppl_k8:.2f} recovery_k8={best_recovery_k8:.1f}% "
        f"fallback={fallback_triggered} params_k5={correction_params_k5_final} params_k8=r{best_rank_k8}:{len(FFN_BLOCKS_K8)*2*D_MODEL*best_rank_k8}"
    )
    f.write(f"117\t133\tcascade_correction_k5_k8_gpt2\t{verdict}\t{tsv_notes}\n")

print(f"Saved manifest: {manifest_path}")
print("\nDone.")
