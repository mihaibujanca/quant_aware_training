"""Cycle 159: Correction-first compression (item 169).

Pipeline A (baseline): compress W_ffn → polish → train LoRA corrections.
Pipeline B (correction-first): train LoRA on compressed model, extract C_b,
  compute Y_target = Y_float - C_b @ x_float, compress Y_target instead of Y_float,
  then apply same corrections.

Hypothesis: compressing the residual (what corrections can't reach) may yield
better PPL than compressing the raw FFN output.

Seeds: [42, 137, 7], 200 OWT eval seqs.
Confirmed if correction-first PPL < 57.
Refuted if PPL > 62 or corrections don't transfer.
Inconclusive if 57-62.
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
SEEDS = [42, 137, 7]
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

CORRECTION_RANK = 8
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
# Low-rank correction module (LoRA-style: x + U @ V @ x)
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

    def effective_matrix(self) -> torch.Tensor:
        """Returns the effective correction matrix C = U @ V, shape (d, d)."""
        return self.U @ self.V  # (d, r) @ (r, d) = (d, d)


class CorrectedModel(nn.Module):
    """Wraps a compressed GPT-2 with per-block low-rank corrections applied via hooks."""

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
        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{n_steps}: loss={loss.item():.4f}")

    corrected_model.eval()
    return losses


def _eval_corrected(corrected_model: CorrectedModel, teacher: TeacherBundle, corpus) -> float:
    corrected_model._install_hooks()
    corrected_model.eval()
    try:
        metrics = _evaluate_model(teacher, corrected_model.base_model, corpus, batch_size=BATCH_SIZE)
    finally:
        corrected_model._remove_hooks()
    return float(metrics["perplexity"])


# ---------------------------------------------------------------------------
# Activation collection helpers for Phase B
# ---------------------------------------------------------------------------
def _collect_block_io(
    model: nn.Module,
    block_id: int,
    tokens: torch.Tensor,
    device: torch.device,
    max_seqs: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect block input and FFN output for a GPT-2 block.

    Returns:
        x_in: (N, d_model) — residual stream entering the block
        y_ffn: (N, d_model) — FFN contribution (block_out - x_in - attn_out)

    We capture the block-level residual by hooking the block input and output.
    Then y_ffn ≈ block_output - block_input (simplification: includes attn too,
    but we use it for the lstsq target fitting which is what matters).

    Actually for the correction-first target we need the block output (post-attn+ffn).
    The correction C_b acts on block output x_block_out.
    Y_target = x_block_out_float - C_b @ x_block_out_float
             = (I - C_b) @ x_block_out_float
    We compress to produce outputs close to Y_target.
    """
    model.eval()
    block_inputs = []
    block_outputs = []
    n_used = 0

    def hook_in(module, args):
        # args[0] is the hidden state entering the block
        if isinstance(args, tuple) and len(args) > 0:
            block_inputs.append(args[0].detach().cpu().float())

    def hook_out(module, input, output):
        if isinstance(output, tuple):
            block_outputs.append(output[0].detach().cpu().float())
        else:
            block_outputs.append(output.detach().cpu().float())

    block = model.transformer.h[block_id]
    h_in = block.register_forward_pre_hook(hook_in)
    h_out = block.register_forward_hook(hook_out)

    with torch.no_grad():
        for i in range(min(max_seqs, tokens.shape[0])):
            seq = tokens[i:i+1].to(device)
            model(seq)
            n_used += 1

    h_in.remove()
    h_out.remove()

    if not block_inputs or not block_outputs:
        raise RuntimeError(f"No activations collected for block {block_id}")

    x_in = torch.cat(block_inputs, dim=1).reshape(-1, D_MODEL)   # (N*T, d)
    x_out = torch.cat(block_outputs, dim=1).reshape(-1, D_MODEL)  # (N*T, d)
    return x_in, x_out


def _collect_ffn_io(
    model: nn.Module,
    block_id: int,
    tokens: torch.Tensor,
    device: torch.device,
    max_seqs: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect FFN input activations and output for a specific block.

    Returns:
        ffn_in: (N, d_model) — input to the FFN (after LayerNorm)
        ffn_out: (N, d_model) — output of the FFN (c_proj output, before adding residual)
    """
    model.eval()
    ffn_inputs = []
    ffn_outputs = []

    block = model.transformer.h[block_id]
    mlp = block.mlp

    def hook_ffn_in(module, args):
        if isinstance(args, tuple) and len(args) > 0:
            ffn_inputs.append(args[0].detach().cpu().float())

    def hook_ffn_out(module, input, output):
        ffn_outputs.append(output.detach().cpu().float())

    h_in = mlp.register_forward_pre_hook(hook_ffn_in)
    h_out = mlp.register_forward_hook(hook_ffn_out)

    with torch.no_grad():
        for i in range(min(max_seqs, tokens.shape[0])):
            seq = tokens[i:i+1].to(device)
            model(seq)

    h_in.remove()
    h_out.remove()

    if not ffn_inputs or not ffn_outputs:
        raise RuntimeError(f"No FFN activations for block {block_id}")

    x = torch.cat(ffn_inputs, dim=1).reshape(-1, D_MODEL)
    y = torch.cat(ffn_outputs, dim=1).reshape(-1, D_MODEL)
    return x, y


def _fit_compressed_ffn_to_target(
    atlas: FfnGeometryAtlas,
    budget: CompressionBudgetSpec,
    block_id: int,
    x_probe: torch.Tensor,
    y_target: torch.Tensor,
) -> CompressedFfnLayer:
    """Fit a compressed FFN layer to approximate y_target given x_probe.

    Uses the same topk selection from atlas but refits the output projection
    (lstsq) to approximate y_target instead of original FFN output.

    x_probe: (N, d_model)
    y_target: (N, d_model)
    """
    # Find geometry for this block
    layer = None
    for lg in atlas.layer_geometries:
        if lg.layer_id == block_id:
            layer = lg
            break
    if layer is None:
        raise ValueError(f"No geometry for block {block_id}")

    k = _layer_target_hidden(layer, budget)
    selected = _select_hidden_units(layer, budget, k)

    source_w_in = _standard_weight(layer.handle.c_fc).to(dtype=torch.float32)
    source_b_in = _standard_bias(layer.handle.c_fc)
    use_output_bias = _standard_bias(layer.handle.c_proj) is not None

    student_w_in = source_w_in[selected].contiguous()
    student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()

    # Compute hidden activations using the original W_in on x_probe
    # GPT-2 uses GELU. The hidden sample from atlas may differ in scale,
    # so recompute on x_probe for the new target.
    with torch.no_grad():
        # x_probe is FFN input (post-LN), student_w_in is (k, d_model)
        pre_act = F.linear(x_probe, student_w_in, student_b_in)  # (N, k)
        # GPT-2 uses GELU
        hidden_new = F.gelu(pre_act)  # (N, k)

    # Fit output projection to y_target
    student_w_out, student_b_out = _fit_output_projection(hidden_new, y_target.to(dtype=torch.float32), use_bias=use_output_bias)

    source_param_count = _param_count(source_w_in, source_b_in) + _param_count(
        _standard_weight(layer.handle.c_proj), _standard_bias(layer.handle.c_proj)
    )
    compressed_param_count = _param_count(student_w_in, student_b_in) + _param_count(
        student_w_out, student_b_out
    )

    return CompressedFfnLayer(
        layer_id=block_id,
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
        notes={"selection_mode": budget.selection_mode, "target": "correction_first"},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading GPT-2 teacher...")
load_spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
teacher = load_nanochat_teacher(load_spec)
teacher.model.eval()
print(f"  Teacher loaded on device={teacher.device}")
device = teacher.device

budget = CompressionBudgetSpec(
    hidden_fraction=HIDDEN_FRACTION,
    min_hidden=32,
    selection_mode="topk",
)

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

    # Float baseline
    bl_metrics = _evaluate_model(teacher, teacher.model, corpus, batch_size=BATCH_SIZE)
    baseline_ppl = float(bl_metrics["perplexity"])
    print(f"  Float PPL: {baseline_ppl:.3f}")
    assert 25.0 <= baseline_ppl <= 65.0, f"Float PPL {baseline_ppl} out of range"

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

    # -----------------------------------------------------------------------
    # Phase A: Build standard compressed model
    # -----------------------------------------------------------------------
    print("  Phase A: Compress (topk+lstsq+polish)...")
    polished_weights: dict[int, dict] = {}
    for block_id in FFN_BLOCKS:
        _set_seed(seed)
        single_pack = _derive_block_pack_from_atlas(atlas, budget, [block_id])
        single_student = swap_compressed_ffns(teacher, single_pack, inplace=False)
        single_student.eval()
        losses = _polish_block(single_student, block_id, polish_tokens, POLISH_STEPS, POLISH_LR, device)
        if losses:
            print(f"    block {block_id}: polish {losses[0]:.4f} → {losses[-1]:.4f}")
        polished_weights[block_id] = _extract_polished_weights(single_student, block_id)
        del single_student

    _set_seed(seed)
    pack_A = _derive_block_pack_from_atlas(atlas, budget, FFN_BLOCKS)
    model_A = swap_compressed_ffns(teacher, pack_A, inplace=False)
    model_A.eval()
    for bid in FFN_BLOCKS:
        if bid in polished_weights and polished_weights[bid]:
            _inject_polished_weights(model_A, bid, polished_weights[bid])

    unc_metrics = _evaluate_model(teacher, model_A, corpus, batch_size=BATCH_SIZE)
    uncorrected_ppl = float(unc_metrics["perplexity"])
    print(f"  Uncorrected PPL (compressed): {uncorrected_ppl:.3f}")

    # Phase A: Train LoRA corrections (r=8)
    print(f"  Phase A: Train LoRA r={CORRECTION_RANK} corrections ({CORRECTION_TRAIN_STEPS} steps)...")
    _set_seed(seed)
    model_A_copy = copy.deepcopy(model_A)
    corrected_A = CorrectedModel(model_A_copy, FFN_BLOCKS, CORRECTION_RANK, D_MODEL)
    corrected_A._install_hooks()
    corrected_A = corrected_A.to(device)
    loss_curve_A = _train_corrections(corrected_A, polish_tokens, CORRECTION_TRAIN_STEPS, CORRECTION_LR, device, seed)
    ppl_A = _eval_corrected(corrected_A, teacher, corpus)
    print(f"  Phase A PPL (compress+correct): {ppl_A:.3f}")

    # Validate Phase A fidelity
    if not (55.0 <= ppl_A <= 65.0):
        print(f"  WARNING: Phase A PPL {ppl_A:.3f} outside expected range [55, 65]")

    # -----------------------------------------------------------------------
    # Extract correction matrices C_b for each block
    # -----------------------------------------------------------------------
    print("  Extracting correction matrices C_b...")
    C_matrices: dict[int, torch.Tensor] = {}
    C_norms: dict[int, float] = {}
    for bid in FFN_BLOCKS:
        lrc = corrected_A.corrections[str(bid)]
        C_b = lrc.effective_matrix().detach().float()  # (d, d) = U @ V
        C_matrices[bid] = C_b.cpu()
        C_norms[bid] = float(C_b.norm().item())
        print(f"    block {bid}: ||C_b|| = {C_norms[bid]:.4f}")
        assert C_norms[bid] > 0.01, f"Block {bid} correction norm too small: {C_norms[bid]}"

    # -----------------------------------------------------------------------
    # Phase A diagnostic: cosine similarity between compressed and float residuals
    # -----------------------------------------------------------------------
    print("  Phase A diagnostic: computing cosine similarities...")
    cosine_sims: dict[int, float] = {}
    for bid in FFN_BLOCKS:
        # Collect block outputs from float model
        _, x_out_float = _collect_block_io(teacher.model, bid, polish_tokens, device, max_seqs=30)
        # Collect block outputs from compressed model (model_A without corrections)
        _, x_out_compressed = _collect_block_io(model_A, bid, polish_tokens, device, max_seqs=30)

        # Residuals: float_out - compressed_out
        res_float = x_out_float - x_out_compressed  # how much we need to add
        # The correction C_b acts on block output to predict this residual
        # cos_sim between what correction sees vs what it's trying to correct
        cos_sim = float(F.cosine_similarity(res_float.flatten().unsqueeze(0),
                                             x_out_compressed.flatten().unsqueeze(0)).item())
        cosine_sims[bid] = cos_sim
        print(f"    block {bid}: cos_sim(compressed_residual, float_residual) = {cos_sim:.4f}")

    min_cosine = min(cosine_sims.values())
    ood_triggered = min_cosine < 0.5
    if ood_triggered:
        print(f"  OOD TRIGGERED: min cosine = {min_cosine:.4f} < 0.5")
    else:
        print(f"  OOD check passed (min cosine = {min_cosine:.4f})")

    # -----------------------------------------------------------------------
    # Phase B: Correction-first compression
    # -----------------------------------------------------------------------
    print("  Phase B: Correction-first compression...")

    # For each block, compute Y_target = Y_ffn_float - C_b @ x_ffn_float
    # The idea: after compression and applying corrections, output ≈ Y_ffn_float.
    # If we compress W such that W @ x ≈ Y_ffn_float - C_b @ x = Y_target,
    # then compressed_output + correction_output = W@x + C_b@x ≈ Y_target + C_b@x = Y_ffn_float.
    # But corrections are applied on block output (residual stream), not FFN output.
    # Correction: x_block_out + C_b @ x_block_out = (I + C_b) @ x_block_out
    # So we want compressed block output x_b such that (I + C_b) @ x_b ≈ x_block_float_out
    # => x_b ≈ (I + C_b)^{-1} @ x_block_float_out ≈ (I - C_b) @ x_block_float_out (first order)
    #
    # For FFN specifically: block_out = block_in + attn_out + ffn_out
    # correction acts on block_out: y = (I+C_b) @ block_out
    # Target for compression: produce ffn_out_target such that
    # (I+C_b) @ (block_in + attn_out + ffn_out_target) ≈ x_block_float_out
    # ffn_out_target ≈ (I+C_b)^{-1} @ x_block_float_out - block_in - attn_out
    # ≈ (I-C_b) @ x_block_float_out - block_in - attn_out (first order approx)
    #
    # Simpler: we target the FFN output directly.
    # Float FFN output: y_ffn_float (collected via hook)
    # Y_target = y_ffn_float - C_b @ (block_out_float)
    # But since C_b norm is small, first order: (I+C_b)^{-1} ≈ I - C_b
    # So Y_target = y_ffn_float - C_b @ block_in  (block_in ≈ block_out for small corrections)
    # We use block_out_float as the correction point:
    # Y_target = y_ffn_float - C_b @ (block_in_float + y_ffn_float)
    # (i.e., adjust ffn_out so that when correction is applied to full block_out, result = float_out)

    cf_layers = []
    lstsq_residual_ratios: dict[int, float] = {}
    probe_tokens = polish_tokens

    for bid in FFN_BLOCKS:
        print(f"    Computing correction-first target for block {bid}...")
        C_b = C_matrices[bid].to(device)  # (d, d)

        # Collect float FFN I/O
        x_ffn_float, y_ffn_float = _collect_ffn_io(teacher.model, bid, probe_tokens, device, max_seqs=40)
        x_ffn_float = x_ffn_float.to(device)
        y_ffn_float = y_ffn_float.to(device)

        # Collect block I/O to get block_in
        x_block_in_float, x_block_out_float = _collect_block_io(teacher.model, bid, probe_tokens, device, max_seqs=40)
        x_block_in_float = x_block_in_float.to(device)
        x_block_out_float = x_block_out_float.to(device)

        # Trim to same size
        n = min(x_ffn_float.shape[0], x_block_in_float.shape[0])
        y_ffn_float = y_ffn_float[:n]
        x_block_out_float = x_block_out_float[:n]
        x_ffn_float = x_ffn_float[:n]

        # First-order approximation:
        # Y_target = y_ffn_float - C_b @ (x_block_in_float + y_ffn_float)
        # = y_ffn_float - C_b @ x_block_out_float  (approx, ignoring attn_out)
        # Note: block_out = block_in + attn_out + ffn_out
        # We use x_block_out_float as the correction point
        with torch.no_grad():
            # C_b: (d, d), x_block_out_float: (N, d) → C_b @ x.T = (d, N), transpose → (N, d)
            correction_contribution = (C_b @ x_block_out_float.T).T  # (N, d)
            y_target = y_ffn_float - correction_contribution

        # Baseline lstsq residual (original y_ffn_float)
        # Collect compressed FFN I/O
        x_ffn_compressed, y_ffn_compressed = _collect_ffn_io(model_A, bid, probe_tokens, device, max_seqs=40)
        x_ffn_compressed = x_ffn_compressed[:n].to(device)
        y_ffn_compressed = y_ffn_compressed[:n].to(device)

        baseline_residual = float((y_ffn_compressed - y_ffn_float).norm().item()) / float(y_ffn_float.norm().item() + 1e-8)

        # Fit compressed FFN to y_target using x_ffn_float as input
        layer_geom = None
        for lg in atlas.layer_geometries:
            if lg.layer_id == bid:
                layer_geom = lg
                break

        k_target = _layer_target_hidden(layer_geom, budget)
        selected = _select_hidden_units(layer_geom, budget, k_target)
        source_w_in = _standard_weight(layer_geom.handle.c_fc).to(dtype=torch.float32).to(device)
        source_b_in_raw = _standard_bias(layer_geom.handle.c_fc)
        source_b_in = source_b_in_raw.to(device) if source_b_in_raw is not None else None
        use_output_bias = _standard_bias(layer_geom.handle.c_proj) is not None

        student_w_in = source_w_in[selected].contiguous()
        student_b_in = None if source_b_in is None else source_b_in[selected].contiguous()

        # Compute activations on the float FFN input
        with torch.no_grad():
            pre_act = F.linear(x_ffn_float[:n], student_w_in, student_b_in)
            hidden_act = F.gelu(pre_act)

        # Fit output projection to y_target
        student_w_out, student_b_out = _fit_output_projection(
            hidden_act.cpu().float(), y_target.cpu().float(), use_bias=use_output_bias
        )
        student_w_out = student_w_out.to(device)
        student_b_out = student_b_out.to(device) if student_b_out is not None else None

        # Compute cf lstsq residual
        with torch.no_grad():
            y_cf_pred = F.linear(hidden_act, student_w_out, student_b_out)
            cf_residual = float((y_cf_pred.cpu() - y_target.cpu()).norm().item()) / float(y_target.norm().item() + 1e-8)

        ratio = cf_residual / (baseline_residual + 1e-8)
        lstsq_residual_ratios[bid] = ratio
        print(f"    block {bid}: baseline_rel_residual={baseline_residual:.4f}, cf_rel_residual={cf_residual:.4f}, ratio={ratio:.3f}")

        source_w_in_cpu = _standard_weight(layer_geom.handle.c_fc).to(dtype=torch.float32)
        source_b_in_cpu = _standard_bias(layer_geom.handle.c_fc)
        source_param_count = _param_count(source_w_in_cpu, source_b_in_cpu) + _param_count(
            _standard_weight(layer_geom.handle.c_proj), _standard_bias(layer_geom.handle.c_proj)
        )
        compressed_param_count = _param_count(student_w_in.cpu(), student_b_in.cpu() if student_b_in is not None else None) + _param_count(
            student_w_out.cpu(), student_b_out.cpu() if student_b_out is not None else None
        )

        cf_layer = CompressedFfnLayer(
            layer_id=bid,
            activation_kind=layer_geom.activation_kind,
            source_hidden_dim=layer_geom.hidden_dim,
            compressed_hidden_dim=int(k_target),
            c_fc_weight=student_w_in.cpu(),
            c_fc_bias=student_b_in.cpu() if student_b_in is not None else None,
            c_proj_weight=student_w_out.cpu(),
            c_proj_bias=student_b_out.cpu() if student_b_out is not None else None,
            selected_indices=selected,
            source_param_count=source_param_count,
            compressed_param_count=compressed_param_count,
            notes={"selection_mode": budget.selection_mode, "target": "correction_first"},
        )
        cf_layers.append(cf_layer)

    # Check lstsq residual gate
    max_ratio = max(lstsq_residual_ratios.values())
    if max_ratio > 2.0:
        print(f"  INCONCLUSIVE gate triggered: max lstsq ratio = {max_ratio:.3f} > 2.0")

    # Build correction-first model (Phase B)
    print("  Building correction-first compressed model...")
    cf_pack = CompressedFfnPack(
        method="geometric",
        model_family=atlas.model_family,
        dataset_name=atlas.dataset_name,
        layers=tuple(cf_layers),
        metadata={"sequence_length": atlas.sequence_length, "pipeline": "correction_first"},
    )
    model_B = swap_compressed_ffns(teacher, cf_pack, inplace=False)
    model_B.eval()

    # Apply same corrections (from Phase A) to correction-first model
    print(f"  Applying Phase A corrections to correction-first model...")
    _set_seed(seed)
    corrected_B = CorrectedModel(model_B, FFN_BLOCKS, CORRECTION_RANK, D_MODEL)
    # Copy trained correction weights from Phase A
    with torch.no_grad():
        for bid in FFN_BLOCKS:
            src = corrected_A.corrections[str(bid)]
            dst = corrected_B.corrections[str(bid)]
            dst.U.data.copy_(src.U.data)
            dst.V.data.copy_(src.V.data)

    corrected_B._install_hooks()
    corrected_B.eval()
    corrected_B = corrected_B.to(device)
    ppl_B_transferred = _eval_corrected(corrected_B, teacher, corpus)
    print(f"  Phase B PPL (correction-first, transferred corrections): {ppl_B_transferred:.3f}")

    # Also: uncorrected Phase B
    unc_B_metrics = _evaluate_model(teacher, model_B, corpus, batch_size=BATCH_SIZE)
    ppl_B_uncorrected = float(unc_B_metrics["perplexity"])
    print(f"  Phase B PPL (correction-first, no corrections): {ppl_B_uncorrected:.3f}")

    # Optionally retrain corrections on model_B for fair comparison
    print(f"  Retraining corrections on correction-first model B...")
    _set_seed(seed)
    model_B_copy = copy.deepcopy(model_B)
    corrected_B_retrained = CorrectedModel(model_B_copy, FFN_BLOCKS, CORRECTION_RANK, D_MODEL)
    corrected_B_retrained._install_hooks()
    corrected_B_retrained = corrected_B_retrained.to(device)
    loss_curve_B = _train_corrections(corrected_B_retrained, polish_tokens, CORRECTION_TRAIN_STEPS, CORRECTION_LR, device, seed)
    ppl_B_retrained = _eval_corrected(corrected_B_retrained, teacher, corpus)
    print(f"  Phase B PPL (correction-first, retrained corrections): {ppl_B_retrained:.3f}")

    per_seed[seed] = {
        "baseline_ppl": baseline_ppl,
        "uncorrected_ppl_A": uncorrected_ppl,
        "ppl_A_corrected": ppl_A,
        "ppl_B_transferred": ppl_B_transferred,
        "ppl_B_uncorrected": ppl_B_uncorrected,
        "ppl_B_retrained": ppl_B_retrained,
        "cosine_sims": cosine_sims,
        "C_norms": C_norms,
        "lstsq_residual_ratios": lstsq_residual_ratios,
        "max_lstsq_ratio": max_ratio,
        "ood_triggered": ood_triggered,
        "loss_curve_A_start": float(loss_curve_A[0]),
        "loss_curve_A_end": float(loss_curve_A[-1]),
        "loss_curve_B_start": float(loss_curve_B[0]),
        "loss_curve_B_end": float(loss_curve_B[-1]),
    }

    del corrected_A, corrected_B, corrected_B_retrained, model_A, model_B, model_A_copy, model_B_copy

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
print(f"\n{'='*60}\nAGGREGATING\n{'='*60}")


def _agg(key):
    vals = [per_seed[s][key] for s in SEEDS]
    return float(np.mean(vals)), float(np.std(vals))


mean_bl, std_bl = _agg("baseline_ppl")
mean_unc_A, std_unc_A = _agg("uncorrected_ppl_A")
mean_A, std_A = _agg("ppl_A_corrected")
mean_B_trans, std_B_trans = _agg("ppl_B_transferred")
mean_B_unc, std_B_unc = _agg("ppl_B_uncorrected")
mean_B_ret, std_B_ret = _agg("ppl_B_retrained")

print(f"Float baseline:              PPL = {mean_bl:.3f} ± {std_bl:.3f}")
print(f"Uncorrected (compress only): PPL = {mean_unc_A:.3f} ± {std_unc_A:.3f}")
print(f"Phase A (compress+correct):  PPL = {mean_A:.3f} ± {std_A:.3f}")
print(f"Phase B uncorrected:         PPL = {mean_B_unc:.3f} ± {std_B_unc:.3f}")
print(f"Phase B transferred corr:    PPL = {mean_B_trans:.3f} ± {std_B_trans:.3f}")
print(f"Phase B retrained corr:      PPL = {mean_B_ret:.3f} ± {std_B_ret:.3f}")

# Key metric: best correction-first PPL vs Phase A
best_B_ppl = min(mean_B_trans, mean_B_ret)
best_B_label = "transferred" if mean_B_trans < mean_B_ret else "retrained"
improvement_vs_A = mean_A - best_B_ppl
print(f"\nBest correction-first: {best_B_ppl:.3f} ({best_B_label})")
print(f"Improvement vs Phase A: {improvement_vs_A:+.3f}")

# Check lstsq gate: aggregate
all_ratios = {}
for bid in FFN_BLOCKS:
    ratios = [per_seed[s]["lstsq_residual_ratios"][bid] for s in SEEDS]
    all_ratios[bid] = float(np.mean(ratios))
max_mean_ratio = max(all_ratios.values())

# Verdict
if max_mean_ratio > 2.0:
    verdict = "INCONCLUSIVE"
    verdict_reason = f"lstsq residual ratio {max_mean_ratio:.2f} > 2.0 (compression fitting degraded)"
elif best_B_ppl < 57.0:
    verdict = "CONFIRMED"
    verdict_reason = f"correction-first PPL {best_B_ppl:.2f} < 57 (beats baseline by {improvement_vs_A:.2f})"
elif best_B_ppl > 62.0:
    verdict = "REFUTED"
    verdict_reason = f"correction-first PPL {best_B_ppl:.2f} > 62 (worse than baseline {mean_A:.2f})"
else:
    verdict = "INCONCLUSIVE"
    verdict_reason = f"correction-first PPL {best_B_ppl:.2f} in [57, 62] (marginal)"

print(f"\nVerdict: {verdict}")
print(f"Reason: {verdict_reason}")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
results = {
    "cycle": 159,
    "item": 169,
    "seeds": SEEDS,
    "ffn_blocks": FFN_BLOCKS,
    "hidden_fraction": HIDDEN_FRACTION,
    "polish_steps": POLISH_STEPS,
    "correction_rank": CORRECTION_RANK,
    "correction_train_steps": CORRECTION_TRAIN_STEPS,
    "correction_lr": CORRECTION_LR,
    "d_model": D_MODEL,
    "baseline_ppl": {"mean": mean_bl, "std": std_bl},
    "uncorrected_ppl": {"mean": mean_unc_A, "std": std_unc_A},
    "phase_A_ppl": {"mean": mean_A, "std": std_A},
    "phase_B_transferred_ppl": {"mean": mean_B_trans, "std": std_B_trans},
    "phase_B_uncorrected_ppl": {"mean": mean_B_unc, "std": std_B_unc},
    "phase_B_retrained_ppl": {"mean": mean_B_ret, "std": std_B_ret},
    "best_correction_first_ppl": best_B_ppl,
    "best_correction_first_variant": best_B_label,
    "improvement_vs_A": improvement_vs_A,
    "max_lstsq_ratio": max_mean_ratio,
    "per_block_lstsq_ratios": {str(b): all_ratios[b] for b in FFN_BLOCKS},
    "verdict": verdict,
    "verdict_reason": verdict_reason,
    "per_seed": {
        str(s): {k: v for k, v in per_seed[s].items() if not isinstance(v, list)}
        for s in SEEDS
    },
}

json_path = RESULTS_DIR / "cycle_159_correction_first.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved JSON: {json_path}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
labels = ["float", "uncorrected", "Phase A\n(compress+correct)", "Phase B\n(cf, transferred)", "Phase B\n(cf, retrained)"]
means = [mean_bl, mean_unc_A, mean_A, mean_B_trans, mean_B_ret]
stds = [std_bl, std_unc_A, std_A, std_B_trans, std_B_ret]
colors = ["#4CAF50", "#F44336", "#2196F3", "#FF9800", "#9C27B0"]
x = np.arange(len(labels))
bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Perplexity")
ax.set_title(f"Correction-First Compression — GPT-2 k=4 6x r={CORRECTION_RANK}\nVerdict: {verdict} | best_B={best_B_ppl:.2f}, A={mean_A:.2f}")
ax.axhline(mean_bl, color="#4CAF50", linestyle="--", alpha=0.5, label="float baseline")
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 0.3, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
ax.legend()
plt.tight_layout()
png_path = RESULTS_DIR / "cycle_159_correction_first.png"
plt.savefig(png_path, dpi=120)
plt.close()
print(f"Saved plot: {png_path}")

print("\nDone.")
