"""Cycle 125: Correction placement sweep — GPT-2 FFN k=4 (item 142).

5 placements tested at equal param budget (12,288 params/block, 49,152 total):
  P1: after FFN, before attention (residual stream, cycle 115 baseline) — r=8
  P2: after attention, before ln_2 — r=8
  P3: Q+K projections (additive, hook c_attn output) — r=4 each
  P4: V+O projections — r=4 each
  P5: P1+P2 combined — r=4 each

All: 2000 steps, lr=1e-4, seed 42. Escalate to seeds [42, 123, 7] if any
non-P1 beats P1 by >=5% relative PPL.

Verdict criteria:
  CONFIRMED    if any non-P1 beats P1 by >=5% relative PPL
  REFUTED      if P1 is within 2% of best alternative
  INCONCLUSIVE if 2-5% differences or ranking not robust
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
SEEDS_INITIAL = [42]
SEEDS_ESCALATION = [42, 123, 7]
EVAL_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4

FFN_BLOCKS = [7, 5, 4, 1]  # k=4
D_MODEL = 768

HIDDEN_FRACTION = 1 / 6
POLISH_STEPS = 500
POLISH_LR = 1e-4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200

CORRECTION_TRAIN_STEPS = 2000
CORRECTION_LR = 1e-4

# Param budget: 12,288 per block = 2 * 768 * 8
BUDGET_PER_BLOCK = 2 * D_MODEL * 8  # 12,288
PLACEMENTS = ["P1", "P2", "P3", "P4", "P5"]

# Cycle 115 seed-42 P1 reference PPL
CYCLE115_P1_SEED42 = 69.53

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


def _compress_blocks(
    teacher: TeacherBundle,
    atlas: FfnGeometryAtlas,
    budget: CompressionBudgetSpec,
    block_ids: list[int],
    seed: int,
    corpus,
) -> nn.Module:
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
# Low-rank correction modules
# ---------------------------------------------------------------------------

class LowRankCorrection(nn.Module):
    """Additive low-rank perturbation: out = x + U @ (V @ x).
    U: (d, r), V: (r, d). U initialized to zero => identity at start.
    """
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(d_model, rank))
        self.V = nn.Parameter(torch.randn(rank, d_model) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (x @ self.V.t()) @ self.U.t()

    @property
    def n_params(self) -> int:
        return self.U.numel() + self.V.numel()


# ---------------------------------------------------------------------------
# Placement model wrappers
# ---------------------------------------------------------------------------

class PlacementModel(nn.Module):
    """Base class for placement experiments. Subclasses install different hooks."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self._hooks: list = []

    def install_hooks(self) -> None:
        self.remove_hooks()
        self._install_hooks_impl()

    def _install_hooks_impl(self) -> None:
        raise NotImplementedError

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, input_ids: torch.Tensor):
        return self.base_model(input_ids)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.corrections.parameters())

    def __del__(self):
        self.remove_hooks()


class P1Model(PlacementModel):
    """P1: correction after full transformer block (residual stream after FFN).
    Hook: block.register_forward_hook. r=8.
    """
    def __init__(self, base_model: nn.Module, block_ids: list[int], rank: int = 8, d_model: int = 768):
        super().__init__(base_model)
        self.block_ids = sorted(block_ids)
        self.corrections = nn.ModuleDict({
            str(bid): LowRankCorrection(d_model, rank) for bid in block_ids
        })

    def _install_hooks_impl(self) -> None:
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

            self._hooks.append(block.register_forward_hook(make_hook(correction)))


class P2Model(PlacementModel):
    """P2: correction after attention sub-layer, before ln_2.
    Hook: pre-hook on block.ln_2 that applies correction to its input.
    r=8.
    """
    def __init__(self, base_model: nn.Module, block_ids: list[int], rank: int = 8, d_model: int = 768):
        super().__init__(base_model)
        self.block_ids = sorted(block_ids)
        self.corrections = nn.ModuleDict({
            str(bid): LowRankCorrection(d_model, rank) for bid in block_ids
        })

    def _install_hooks_impl(self) -> None:
        for bid in self.block_ids:
            correction = self.corrections[str(bid)]
            block = self.base_model.transformer.h[bid]

            def make_pre_hook(corr):
                def hook(module, args):
                    # args is a tuple of inputs to ln_2
                    x = args[0]
                    x_corrected = corr(x)
                    return (x_corrected,) + args[1:]
                return hook

            self._hooks.append(block.ln_2.register_forward_pre_hook(make_pre_hook(correction)))


class P3Model(PlacementModel):
    """P3: corrections on Q+K projections via c_attn hook.
    GPT-2 has fused c_attn (768->2304). We hook its output and add
    low-rank corrections to the Q [:,:,:768] and K [:,:,768:1536] segments.
    To avoid in-place autograd issues, we build a correction tensor and add.
    r=4 each (same total budget as r=8 in P1).
    """
    def __init__(self, base_model: nn.Module, block_ids: list[int], rank: int = 4, d_model: int = 768):
        super().__init__(base_model)
        self.block_ids = sorted(block_ids)
        # Two corrections per block: q_corr and k_corr
        self.corrections = nn.ModuleDict()
        for bid in block_ids:
            self.corrections[f"{bid}_q"] = LowRankCorrection(d_model, rank)
            self.corrections[f"{bid}_k"] = LowRankCorrection(d_model, rank)

    def _install_hooks_impl(self) -> None:
        for bid in self.block_ids:
            q_corr = self.corrections[f"{bid}_q"]
            k_corr = self.corrections[f"{bid}_k"]
            block = self.base_model.transformer.h[bid]
            d = q_corr.U.shape[0]  # 768

            def make_hook(qc, kc, d_model):
                def hook(module, input, output):
                    # output: (batch, seq, 2304) — Q|K|V concatenated
                    # input[0]: (batch, seq, 768) — ln_1 output
                    x = input[0]  # the input to c_attn
                    q_delta = qc(x) - x  # low-rank delta only: shape (B, S, d)
                    k_delta = kc(x) - x
                    # Build full-width correction tensor (avoids in-place on views)
                    correction = torch.cat([
                        q_delta,
                        k_delta,
                        torch.zeros_like(output[:, :, 2*d_model:])
                    ], dim=-1)
                    return output + correction
                return hook

            self._hooks.append(block.attn.c_attn.register_forward_hook(make_hook(q_corr, k_corr, d)))


class P4Model(PlacementModel):
    """P4: corrections on V+O projections.
    V: hook c_attn output, correct the V segment [:,:,1536:2304].
    O: hook c_proj output, correct the attention output.
    r=4 each.
    """
    def __init__(self, base_model: nn.Module, block_ids: list[int], rank: int = 4, d_model: int = 768):
        super().__init__(base_model)
        self.block_ids = sorted(block_ids)
        self.corrections = nn.ModuleDict()
        for bid in block_ids:
            self.corrections[f"{bid}_v"] = LowRankCorrection(d_model, rank)
            self.corrections[f"{bid}_o"] = LowRankCorrection(d_model, rank)

    def _install_hooks_impl(self) -> None:
        for bid in self.block_ids:
            v_corr = self.corrections[f"{bid}_v"]
            o_corr = self.corrections[f"{bid}_o"]
            block = self.base_model.transformer.h[bid]
            d = v_corr.U.shape[0]  # 768

            # V correction: hook c_attn output
            def make_v_hook(vc, d_model):
                def hook(module, input, output):
                    x = input[0]  # ln_1 output
                    v_delta = vc(x) - x  # (B, S, d)
                    correction = torch.cat([
                        torch.zeros_like(output[:, :, :2*d_model]),
                        v_delta
                    ], dim=-1)
                    return output + correction
                return hook

            self._hooks.append(block.attn.c_attn.register_forward_hook(make_v_hook(v_corr, d)))

            # O correction: hook c_proj output (attention projection)
            def make_o_hook(oc):
                def hook(module, input, output):
                    # input[0]: attention context after value weighting, (B, S, d)
                    x = input[0]
                    o_delta = oc(x) - x
                    return output + o_delta
                return hook

            self._hooks.append(block.attn.c_proj.register_forward_hook(make_o_hook(o_corr)))


class P5Model(PlacementModel):
    """P5: P1+P2 combined (r=4 each, same total budget).
    One correction after FFN (P1 position) and one after attention (P2 position).
    """
    def __init__(self, base_model: nn.Module, block_ids: list[int], rank: int = 4, d_model: int = 768):
        super().__init__(base_model)
        self.block_ids = sorted(block_ids)
        self.corrections = nn.ModuleDict()
        for bid in block_ids:
            self.corrections[f"{bid}_p1"] = LowRankCorrection(d_model, rank)
            self.corrections[f"{bid}_p2"] = LowRankCorrection(d_model, rank)

    def _install_hooks_impl(self) -> None:
        for bid in self.block_ids:
            p1_corr = self.corrections[f"{bid}_p1"]
            p2_corr = self.corrections[f"{bid}_p2"]
            block = self.base_model.transformer.h[bid]

            # P1: after full block
            def make_block_hook(corr):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        return (corr(h),) + output[1:]
                    return corr(output)
                return hook

            self._hooks.append(block.register_forward_hook(make_block_hook(p1_corr)))

            # P2: before ln_2 (after attention residual)
            def make_ln2_prehook(corr):
                def hook(module, args):
                    x = args[0]
                    return (corr(x),) + args[1:]
                return hook

            self._hooks.append(block.ln_2.register_forward_pre_hook(make_ln2_prehook(p2_corr)))


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def _train_placement(
    model: PlacementModel,
    train_tokens: torch.Tensor,
    n_steps: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> list[float]:
    """Train only the correction parameters under next-token CE loss."""
    model.base_model.requires_grad_(False)
    model.corrections.requires_grad_(True)

    params = list(model.corrections.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    model.base_model.eval()
    model.train()

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
        raw = model(inputs)
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if (step + 1) % 200 == 0:
            print(f"      step {step+1}/{n_steps}: loss={loss.item():.4f}")

    model.eval()
    return losses


def _eval_placement(
    model: PlacementModel,
    teacher: TeacherBundle,
    corpus,
) -> float:
    """Evaluate PPL with hooks active."""
    model.install_hooks()
    model.eval()
    try:
        metrics = _evaluate_model(teacher, model.base_model, corpus, batch_size=BATCH_SIZE)
    finally:
        model.remove_hooks()
    return float(metrics["perplexity"])


def _make_placement_model(
    placement: str,
    base_model: nn.Module,
    block_ids: list[int],
    d_model: int,
) -> PlacementModel:
    if placement == "P1":
        return P1Model(base_model, block_ids, rank=8, d_model=d_model)
    elif placement == "P2":
        return P2Model(base_model, block_ids, rank=8, d_model=d_model)
    elif placement == "P3":
        return P3Model(base_model, block_ids, rank=4, d_model=d_model)
    elif placement == "P4":
        return P4Model(base_model, block_ids, rank=4, d_model=d_model)
    elif placement == "P5":
        return P5Model(base_model, block_ids, rank=4, d_model=d_model)
    else:
        raise ValueError(f"Unknown placement: {placement}")


def _verify_param_budget(model: PlacementModel, placement: str, n_blocks: int) -> None:
    total = model.param_count()
    expected = BUDGET_PER_BLOCK * n_blocks
    per_block = total // n_blocks
    print(f"    {placement} param check: {total:,} total, {per_block:,}/block (expected {BUDGET_PER_BLOCK:,}/block, {expected:,} total)")
    if total != expected:
        print(f"    WARNING: param count mismatch! got {total}, expected {expected}")


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
# Phase 1: seed 42 sweep across all 5 placements
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"PHASE 1: Initial sweep (seed 42), k=4 blocks={FFN_BLOCKS}")
print(f"{'='*60}")

_set_seed(42)
corpus = build_probe_corpus(ProbeCorpusSpec(
    teacher=teacher,
    text_path=str(OWT_PATH),
    sequence_length=SEQ_LEN,
    probe_train_sequences=PROBE_TRAIN_SEQS,
    probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
    eval_val_sequences=EVAL_SEQS,
    dataset_name="openwebtext",
    shuffle_seed=42,
))

# Baseline float PPL
bl_metrics = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
baseline_ppl = float(bl_metrics["perplexity"])
print(f"  Baseline float PPL: {baseline_ppl:.3f}")

if not (25.0 <= baseline_ppl <= 65.0):
    print(f"ABORT: baseline PPL {baseline_ppl:.2f} outside [25, 65]")
    sys.exit(1)

# Probe and compress
print(f"\n  Probing FFN geometry for blocks {sorted(FFN_BLOCKS)}...")
_set_seed(42)
atlas = probe_ffn_geometry(
    teacher,
    corpus,
    layer_ids=tuple(sorted(FFN_BLOCKS)),
    max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
    batch_size=BATCH_SIZE,
)

compressed_model = _compress_blocks(teacher, atlas, budget, FFN_BLOCKS, 42, corpus)
unc_metrics = _evaluate_model(teacher, compressed_model, corpus, batch_size=BATCH_SIZE)
uncorrected_ppl = float(unc_metrics["perplexity"])
degradation = uncorrected_ppl - baseline_ppl
print(f"  Uncorrected k=4 PPL: {uncorrected_ppl:.3f}  (degradation={degradation:.3f})")

if uncorrected_ppl >= 200.0:
    print(f"ABORT: uncorrected PPL {uncorrected_ppl:.2f} is too high. Compression broken.")
    sys.exit(1)

# Sweep placements
placement_results: dict[str, dict] = {}

for placement in PLACEMENTS:
    print(f"\n  [{placement}] Training 2000 steps, seed=42...")
    _set_seed(42)
    compressed_copy = copy.deepcopy(compressed_model)
    pm = _make_placement_model(placement, compressed_copy, FFN_BLOCKS, D_MODEL)
    _verify_param_budget(pm, placement, len(FFN_BLOCKS))
    pm.install_hooks()
    pm = pm.to(teacher.device)

    loss_curve = _train_placement(
        pm, corpus.probe_holdout,
        n_steps=CORRECTION_TRAIN_STEPS,
        lr=CORRECTION_LR,
        device=teacher.device,
        seed=42,
    )
    print(f"    Train loss: {loss_curve[0]:.4f} -> {loss_curve[-1]:.4f}")

    corrected_ppl = _eval_placement(pm, teacher, corpus)
    residual = corrected_ppl - baseline_ppl
    recovery = (degradation - residual) / degradation * 100.0
    print(f"  [{placement}] PPL={corrected_ppl:.3f}  recovery={recovery:.2f}%")

    placement_results[placement] = {
        "ppl": corrected_ppl,
        "recovery": recovery,
        "loss_curve": loss_curve,
        "param_count": pm.param_count(),
    }
    del pm, compressed_copy

# Validate P1
p1_ppl = placement_results["P1"]["ppl"]
print(f"\n  P1 PPL={p1_ppl:.3f} vs cycle 115 reference={CYCLE115_P1_SEED42:.3f}")
if abs(p1_ppl - CYCLE115_P1_SEED42) > 2.0:
    print(f"  WARNING: P1 PPL {p1_ppl:.3f} deviates by {abs(p1_ppl - CYCLE115_P1_SEED42):.3f} from cycle 115 ({CYCLE115_P1_SEED42:.3f})")
else:
    print(f"  P1 reproduction OK (within 2 PPL)")

# Sanity: all placements must beat uncorrected
for p, r in placement_results.items():
    if r["ppl"] >= uncorrected_ppl:
        print(f"  WARNING: {p} PPL {r['ppl']:.3f} >= uncorrected {uncorrected_ppl:.3f}! Hook may be broken.")

# Check escalation criterion
non_p1_best = min(
    (v["ppl"] for k, v in placement_results.items() if k != "P1"),
    default=float("inf")
)
best_non_p1_name = min(
    (k for k in placement_results if k != "P1"),
    key=lambda k: placement_results[k]["ppl"]
)
p1_relative_improvement = (p1_ppl - non_p1_best) / p1_ppl * 100.0  # positive means non-P1 beats P1
print(f"\n  Best non-P1: {best_non_p1_name}={non_p1_best:.3f}, P1={p1_ppl:.3f}")
print(f"  Non-P1 relative improvement over P1: {p1_relative_improvement:.2f}%")

escalate = p1_relative_improvement >= 5.0

# ---------------------------------------------------------------------------
# Phase 2: escalation (if triggered)
# ---------------------------------------------------------------------------
if escalate:
    print(f"\n{'='*60}")
    print(f"PHASE 2: Escalation triggered! Running top 3 placements with seeds {SEEDS_ESCALATION}")
    print(f"{'='*60}")

    # Top 3 by PPL
    ranked = sorted(placement_results.items(), key=lambda x: x[1]["ppl"])
    top3 = [p for p, _ in ranked[:3]]
    print(f"  Top 3 placements: {top3}")

    escalation_results: dict[str, dict[int, dict]] = {p: {} for p in top3}

    for seed in SEEDS_ESCALATION:
        print(f"\n  --- Seed {seed} ---")
        _set_seed(seed)
        corpus_s = build_probe_corpus(ProbeCorpusSpec(
            teacher=teacher,
            text_path=str(OWT_PATH),
            sequence_length=SEQ_LEN,
            probe_train_sequences=PROBE_TRAIN_SEQS,
            probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
            eval_val_sequences=EVAL_SEQS,
            dataset_name="openwebtext",
            shuffle_seed=seed,
        ))
        bl_s = _evaluate_model(teacher, teacher.model, corpus_s, batch_size=BATCH_SIZE)
        baseline_s = float(bl_s["perplexity"])

        _set_seed(seed)
        atlas_s = probe_ffn_geometry(
            teacher, corpus_s,
            layer_ids=tuple(sorted(FFN_BLOCKS)),
            max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
            batch_size=BATCH_SIZE,
        )
        compressed_s = _compress_blocks(teacher, atlas_s, budget, FFN_BLOCKS, seed, corpus_s)
        unc_s = _evaluate_model(teacher, compressed_s, corpus_s, batch_size=BATCH_SIZE)
        unc_ppl_s = float(unc_s["perplexity"])
        deg_s = unc_ppl_s - baseline_s

        for placement in top3:
            print(f"    [{placement}] seed={seed}...")
            _set_seed(seed)
            compressed_copy_s = copy.deepcopy(compressed_s)
            pm_s = _make_placement_model(placement, compressed_copy_s, FFN_BLOCKS, D_MODEL)
            pm_s.install_hooks()
            pm_s = pm_s.to(teacher.device)

            loss_curve_s = _train_placement(
                pm_s, corpus_s.probe_holdout,
                n_steps=CORRECTION_TRAIN_STEPS,
                lr=CORRECTION_LR,
                device=teacher.device,
                seed=seed,
            )
            corr_ppl_s = _eval_placement(pm_s, teacher, corpus_s)
            rec_s = (deg_s - (corr_ppl_s - baseline_s)) / deg_s * 100.0
            print(f"      PPL={corr_ppl_s:.3f}  recovery={rec_s:.2f}%")

            escalation_results[placement][seed] = {
                "baseline": baseline_s,
                "uncorrected": unc_ppl_s,
                "ppl": corr_ppl_s,
                "recovery": rec_s,
                "loss_curve": loss_curve_s,
            }
            del pm_s, compressed_copy_s
        del compressed_s, atlas_s, corpus_s
else:
    escalation_results = {}
    top3 = []
    print("\n  Escalation not triggered (no non-P1 beats P1 by >=5%).")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nVERDICT\n{'='*60}")
print(f"Seed-42 results (PPL):")
for p in PLACEMENTS:
    r = placement_results[p]
    marker = " <-- best" if r["ppl"] == min(v["ppl"] for v in placement_results.values()) else ""
    print(f"  {p}: {r['ppl']:.3f}  recovery={r['recovery']:.2f}%{marker}")

print(f"  Uncorrected: {uncorrected_ppl:.3f}")
print(f"  Float:       {baseline_ppl:.3f}")

if escalate and escalation_results:
    print(f"\nEscalation results (mean PPL over seeds {SEEDS_ESCALATION}):")
    for p in top3:
        ppls = [escalation_results[p][s]["ppl"] for s in SEEDS_ESCALATION]
        print(f"  {p}: mean={np.mean(ppls):.3f} ± {np.std(ppls):.3f}")

# Final verdict
best_non_p1_ppl = min(placement_results[p]["ppl"] for p in PLACEMENTS if p != "P1")
diff_relative = (p1_ppl - best_non_p1_ppl) / p1_ppl * 100.0

if diff_relative >= 5.0:
    if escalate and escalation_results:
        # Confirm with multi-seed
        top1 = sorted(top3, key=lambda p: np.mean([escalation_results[p][s]["ppl"] for s in SEEDS_ESCALATION]))[0]
        top1_mean = np.mean([escalation_results[top1][s]["ppl"] for s in SEEDS_ESCALATION])
        p1_mean = np.mean([escalation_results["P1"][s]["ppl"] for s in SEEDS_ESCALATION]) if "P1" in escalation_results else p1_ppl
        if (p1_mean - top1_mean) / p1_mean * 100.0 >= 5.0:
            verdict = "CONFIRMED"
            verdict_detail = f"{top1} beats P1 by >=5% relative across seeds"
        else:
            verdict = "INCONCLUSIVE"
            verdict_detail = f"Seed-42 showed >=5% but not confirmed across all seeds"
    else:
        verdict = "CONFIRMED"
        verdict_detail = f"{best_non_p1_name} beats P1 by {diff_relative:.1f}% (seed 42 only)"
elif diff_relative <= 2.0:
    verdict = "REFUTED"
    verdict_detail = f"P1 within 2% of best alternative (diff={diff_relative:.1f}%)"
else:
    verdict = "INCONCLUSIVE"
    verdict_detail = f"2-5% difference ({diff_relative:.1f}%), placement ranking ambiguous"

print(f"\nVERDICT: {verdict}")
print(f"  {verdict_detail}")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
results_json = {
    "cycle": 125,
    "item": 142,
    "ffn_blocks": FFN_BLOCKS,
    "d_model": D_MODEL,
    "budget_per_block": BUDGET_PER_BLOCK,
    "correction_train_steps": CORRECTION_TRAIN_STEPS,
    "correction_lr": CORRECTION_LR,
    "baseline_ppl_seed42": baseline_ppl,
    "uncorrected_ppl_seed42": uncorrected_ppl,
    "cycle115_p1_reference": CYCLE115_P1_SEED42,
    "p1_reproduction_delta": abs(p1_ppl - CYCLE115_P1_SEED42),
    "placements_seed42": {
        p: {
            "ppl": placement_results[p]["ppl"],
            "recovery": placement_results[p]["recovery"],
            "param_count": placement_results[p]["param_count"],
        }
        for p in PLACEMENTS
    },
    "escalation_triggered": escalate,
    "escalation_results": {
        p: {
            str(s): {
                "ppl": escalation_results[p][s]["ppl"],
                "recovery": escalation_results[p][s]["recovery"],
            }
            for s in SEEDS_ESCALATION
        }
        for p in top3
    } if escalate else {},
    "verdict": verdict,
    "verdict_detail": verdict_detail,
    "best_non_p1": best_non_p1_name,
    "p1_vs_best_non_p1_pct": diff_relative,
}

json_path = RESULTS_DIR / "cycle_125_placement_sweep.json"
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# 1. PPL bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ppls = [placement_results[p]["ppl"] for p in PLACEMENTS]
recov = [placement_results[p]["recovery"] for p in PLACEMENTS]
colors_bar = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
x = np.arange(len(PLACEMENTS))
bars = ax.bar(x, ppls, color=colors_bar, alpha=0.85)
ax.axhline(uncorrected_ppl, color="#F44336", linestyle="--", label=f"uncorr {uncorrected_ppl:.2f}", alpha=0.7)
ax.axhline(baseline_ppl, color="#4CAF50", linestyle="--", label=f"float {baseline_ppl:.2f}", alpha=0.7)
ax.axhline(CYCLE115_P1_SEED42, color="#2196F3", linestyle=":", label=f"P1 cyc115 ref {CYCLE115_P1_SEED42:.2f}", alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(PLACEMENTS)
ax.set_ylabel("Perplexity (seed 42)")
ax.set_title(f"Placement Sweep — GPT-2 k=4\nVerdict: {verdict}")
for bar, ppl_v, rec_v in zip(bars, ppls, recov):
    ax.text(bar.get_x() + bar.get_width()/2, ppl_v + 0.3, f"{ppl_v:.2f}\n({rec_v:.0f}%)", ha="center", va="bottom", fontsize=8)
ax.legend(fontsize=8)

# 2. Loss curves
ax2 = axes[1]
for p, col in zip(PLACEMENTS, colors_bar):
    curve = placement_results[p]["loss_curve"]
    ax2.plot(range(1, len(curve)+1), curve, alpha=0.8, label=p, color=col)
ax2.set_title("Correction Training Loss (seed 42)")
ax2.set_xlabel("Step")
ax2.set_ylabel("CE Loss")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "cycle_125_ppl_bar.png"
plt.savefig(plot_path, dpi=120)
plt.close()
print(f"Saved plot: {plot_path}")

# 3. Loss curves separately
fig2, ax3 = plt.subplots(figsize=(10, 4))
for p, col in zip(PLACEMENTS, colors_bar):
    curve = placement_results[p]["loss_curve"]
    ax3.plot(range(1, len(curve)+1), curve, alpha=0.8, label=p, color=col)
ax3.set_title(f"Correction Training Loss Curves — Placement Sweep k=4 (seed 42)")
ax3.set_xlabel("Step")
ax3.set_ylabel("CE Loss")
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
plt.tight_layout()
loss_curves_path = RESULTS_DIR / "cycle_125_loss_curves.png"
plt.savefig(loss_curves_path, dpi=120)
plt.close()
print(f"Saved loss curves: {loss_curves_path}")

print("\nDone.")
