"""Cycle 127: SVD of learned corrections — geometric alignment (item 144).

Extract trained P1 LoRA corrections (r=8, blocks [1,4,5,7]), SVD them,
and measure cosine alignment against three reference subspaces:
  (a) compression error E_l (post-FFN, pre-LayerNorm)
  (b) attention amplification directions
  (c) bump channel directions (cycle 110 protocol)

Null baseline: subspace-vs-subspace max|cos| over 4x4 pairs, 1000 trials.

Verdict criteria:
  CONFIRMED    if corrections align with attention amplification (cos>0.5 top-4 in >=3/4 blocks)
  REFUTED      if corrections align with compression error instead (cos>0.5 with error SVD)
  INCONCLUSIVE if low alignment with both (<0.3)
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
SEED = 42
EVAL_SEQS = 200
SEQ_LEN = 128
BATCH_SIZE = 4

FFN_BLOCKS = [7, 5, 4, 1]  # k=4 compressed blocks
ANALYSIS_BLOCKS = [1, 4, 5, 7]  # sorted for analysis
D_MODEL = 768
D_FFN = 3072
RANK = 8
TOP_K_ALIGN = 4  # top-k singular vectors for alignment

HIDDEN_FRACTION = 1 / 6
POLISH_STEPS = 500
POLISH_LR = 1e-4
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
PROBE_HOLDOUT_SEQS = 200

CORRECTION_TRAIN_STEPS = 2000
CORRECTION_LR = 1e-4

NULL_TRIALS = 1000

# Cycle 125 reference
CYCLE125_P1_SEED42 = 69.52

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


def _param_count(w: torch.Tensor, b: Optional[torch.Tensor]) -> int:
    return w.numel() + (0 if b is None else b.numel())


class LowRankCorrection(nn.Module):
    """Additive low-rank: out = x + U @ (V @ x). U zero-init."""
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(d_model, rank))
        self.V = nn.Parameter(torch.randn(rank, d_model) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (x @ self.V.t()) @ self.U.t()


class P1Model(nn.Module):
    """P1: correction after full transformer block (residual stream after FFN)."""
    def __init__(self, base_model: nn.Module, block_ids: list[int], rank: int = 8, d_model: int = 768):
        super().__init__()
        self.base_model = base_model
        self.block_ids = sorted(block_ids)
        self.corrections = nn.ModuleDict({
            str(bid): LowRankCorrection(d_model, rank) for bid in block_ids
        })
        self._hooks: list = []

    def install_hooks(self) -> None:
        self.remove_hooks()
        for bid in self.block_ids:
            correction = self.corrections[str(bid)]
            block = self.base_model.transformer.h[bid]

            def make_hook(corr):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        return (corr(h),) + output[1:]
                    else:
                        return corr(output)
                return hook

            self._hooks.append(block.register_forward_hook(make_hook(correction)))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, input_ids: torch.Tensor):
        return self.base_model(input_ids)


# ---------------------------------------------------------------------------
# Compression setup (reuse from cycle 125)
# ---------------------------------------------------------------------------
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
            _standard_weight(layer.handle.c_proj), _standard_bias(layer.handle.c_proj))
        compressed_param_count = _param_count(student_w_in, student_b_in) + _param_count(
            student_w_out, student_b_out)
        compressed_layers.append(CompressedFfnLayer(
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
        ))
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


def _polish_block(student_model, block_id, polish_tokens, n_steps, lr, device):
    """Polish W_out of one compressed block under CE loss."""
    if n_steps == 0:
        return []
    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
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


def _extract_polished_weights(student_model, block_id):
    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        return {}
    return {
        "c_proj_weight": mlp.c_proj.weight.detach().clone(),
        "c_proj_bias": (mlp.c_proj.bias.detach().clone()
                        if hasattr(mlp.c_proj, "bias") and mlp.c_proj.bias is not None else None),
    }


def _inject_polished_weights(student_model, block_id, polished):
    mlp = _find_compressed_mlp(student_model, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        return
    with torch.no_grad():
        mlp.c_proj.weight.copy_(polished["c_proj_weight"])
        if polished["c_proj_bias"] is not None and hasattr(mlp.c_proj, "bias") and mlp.c_proj.bias is not None:
            mlp.c_proj.bias.copy_(polished["c_proj_bias"])


def _train_corrections(model, train_tokens, n_steps, lr, device, seed):
    """Train P1 corrections under CE loss."""
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
            print(f"      step {step+1}/{n_steps}: loss={loss.item():.4f}", flush=True)
    model.eval()
    return losses


# ---------------------------------------------------------------------------
# SVD alignment analysis
# ---------------------------------------------------------------------------
def svd_top_k(matrix: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD of matrix, return top-k left singular vectors, singular values, right singular vectors."""
    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k, :]


def cosine_alignment_matrix(vecs_a: torch.Tensor, vecs_b: torch.Tensor) -> torch.Tensor:
    """Compute |cos(a_i, b_j)| for all pairs. vecs_a: (d, k1), vecs_b: (d, k2)."""
    # Normalize
    a_norm = vecs_a / (vecs_a.norm(dim=0, keepdim=True) + 1e-12)
    b_norm = vecs_b / (vecs_b.norm(dim=0, keepdim=True) + 1e-12)
    return (a_norm.T @ b_norm).abs()  # (k1, k2)


def null_baseline_max_cos(d: int, k: int, n_trials: int = 1000) -> dict:
    """Null baseline: max|cos| over kxk pairs from two random rank-k subspaces in R^d."""
    maxcos_list = []
    for _ in range(n_trials):
        A = torch.randn(d, k)
        B = torch.randn(d, k)
        Ua, _, _ = torch.linalg.svd(A, full_matrices=False)
        Ub, _, _ = torch.linalg.svd(B, full_matrices=False)
        align = (Ua.T @ Ub).abs()
        maxcos_list.append(float(align.max()))
    arr = np.array(maxcos_list)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p99": float(np.percentile(arr, 99)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


@torch.no_grad()
def collect_residual_stream(model, tokens, block_ids, device):
    """Collect post-FFN, pre-LayerNorm residual stream activations at specified blocks.
    Hook placement: after block forward (x = x + attn + ffn), before next block's ln_1.
    This is the raw residual stream sum.
    Returns: dict[block_id] -> (n_tokens, d_model) tensor.
    """
    activations = {bid: [] for bid in block_ids}
    hooks = []

    for bid in block_ids:
        block = model.transformer.h[bid]

        def make_hook(block_id):
            def hook(module, input, output):
                # output[0] is the hidden state after the full block (attn + ffn + residuals)
                h = output[0] if isinstance(output, tuple) else output
                activations[block_id].append(h.detach().cpu())
            return hook

        hooks.append(block.register_forward_hook(make_hook(bid)))

    n_seqs = tokens.shape[0]
    for i in range(0, n_seqs, BATCH_SIZE):
        batch = tokens[i:i+BATCH_SIZE].to(device)
        inputs = batch[:, :-1].contiguous()
        model(inputs)

    for h in hooks:
        h.remove()

    result = {}
    for bid in block_ids:
        stacked = torch.cat(activations[bid], dim=0)  # (n_batches * batch_seq, d_model)
        # Flatten to (total_tokens, d_model)
        result[bid] = stacked.reshape(-1, D_MODEL)
    return result


@torch.no_grad()
def collect_attention_output_delta(float_model, compressed_model, tokens, block_ids, device):
    """Collect attention output perturbation (delta) per block.
    Hook on block.attn output to capture the attention sub-layer output difference.
    """
    def _collect_attn_outputs(model, tokens_t, bids):
        attn_outs = {bid: [] for bid in bids}
        hooks = []
        for bid in bids:
            attn = model.transformer.h[bid].attn

            def make_hook(block_id):
                def hook(module, input, output):
                    # GPT-2 attn returns (attn_output, present, (attn_weights))
                    h = output[0] if isinstance(output, tuple) else output
                    attn_outs[block_id].append(h.detach().cpu())
                return hook

            hooks.append(attn.register_forward_hook(make_hook(bid)))

        n_seqs = tokens_t.shape[0]
        for i in range(0, n_seqs, BATCH_SIZE):
            batch = tokens_t[i:i+BATCH_SIZE].to(device)
            inputs = batch[:, :-1].contiguous()
            model(inputs)

        for h in hooks:
            h.remove()

        result = {}
        for bid in bids:
            stacked = torch.cat(attn_outs[bid], dim=0)
            result[bid] = stacked.reshape(-1, D_MODEL)
        return result

    float_attn = _collect_attn_outputs(float_model, tokens, block_ids)
    comp_attn = _collect_attn_outputs(compressed_model, tokens, block_ids)

    deltas = {}
    for bid in block_ids:
        deltas[bid] = float_attn[bid] - comp_attn[bid]
    return deltas


@torch.no_grad()
def compute_bump_directions(model, tokens, block_ids, device):
    """Identify bump-dominated channels (|z| < 1.0 for >50% of tokens) per block.
    Returns axis-aligned indicator vectors as columns.
    """
    bump_dirs = {}
    for bid in block_ids:
        block = model.transformer.h[bid]
        pre_acts = []

        # Hook to capture pre-activation (input to GELU)
        fc = block.mlp.c_fc
        def make_hook():
            def hook(module, input, output):
                pre_acts.append(output.detach().cpu())
            return hook

        h = fc.register_forward_hook(make_hook())
        n_seqs = tokens.shape[0]
        for i in range(0, n_seqs, BATCH_SIZE):
            batch = tokens[i:i+BATCH_SIZE].to(device)
            inputs = batch[:, :-1].contiguous()
            model(inputs)
        h.remove()

        z_all = torch.cat(pre_acts, dim=0).reshape(-1, D_FFN)  # (tokens, 3072)
        bump_frac = (z_all.abs() < 1.0).float().mean(dim=0)  # per channel
        bump_channels = (bump_frac > 0.5).nonzero(as_tuple=True)[0]

        # Project bump channels to output space via W_down
        # Bump directions in residual stream = columns of W_down for bump channels
        w_down = _standard_weight(block.mlp.c_proj).float()  # (768, 3072)
        if len(bump_channels) > 0:
            dirs = w_down[:, bump_channels]  # (768, n_bump)
            # Normalize columns
            dirs = dirs / (dirs.norm(dim=0, keepdim=True) + 1e-12)
            # Take SVD to get top-k orthogonal directions
            if dirs.shape[1] > TOP_K_ALIGN:
                U_bump, _, _ = torch.linalg.svd(dirs, full_matrices=False)
                dirs = U_bump[:, :TOP_K_ALIGN]
        else:
            dirs = torch.zeros(D_MODEL, TOP_K_ALIGN)

        bump_dirs[bid] = dirs
        print(f"    Block {bid}: {len(bump_channels)} bump channels (>{0.5*100:.0f}% in |z|<1)", flush=True)

    return bump_dirs


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    _set_seed(SEED)

    # 1. Load teacher
    print("Loading GPT-2 teacher...", flush=True)
    teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
    print(f"  Teacher loaded on device={teacher.device}", flush=True)

    # Keep a copy of the float model for reference comparisons
    float_model = teacher.model
    float_model.eval()

    # Build corpus
    corpus = build_probe_corpus(ProbeCorpusSpec(
        teacher=teacher,
        text_path=str(OWT_PATH),
        sequence_length=SEQ_LEN,
        probe_train_sequences=PROBE_TRAIN_SEQS,
        probe_holdout_sequences=PROBE_HOLDOUT_SEQS,
        eval_val_sequences=EVAL_SEQS,
        dataset_name="openwebtext",
        shuffle_seed=SEED,
    ))

    # Baseline float PPL
    bl_metrics = _evaluate_model(teacher, float_model, corpus, batch_size=BATCH_SIZE)
    print(f"  Float PPL: {bl_metrics['perplexity']:.3f}", flush=True)

    # 2. Compress k=4 blocks (exact cycle 125 pattern)
    print(f"\nCompressing blocks {FFN_BLOCKS}...", flush=True)
    budget = CompressionBudgetSpec(hidden_fraction=HIDDEN_FRACTION)
    atlas = probe_ffn_geometry(
        teacher, corpus,
        layer_ids=tuple(sorted(FFN_BLOCKS)),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )

    # Per-block polish then assemble
    polish_tokens = corpus.probe_holdout
    print(f"  Polishing {POLISH_STEPS} steps per block...", flush=True)
    polished_weights: dict[int, dict] = {}
    for bid in FFN_BLOCKS:
        _set_seed(SEED)
        single_pack = _derive_block_pack_from_atlas(atlas, budget, [bid])
        single_student = swap_compressed_ffns(teacher, single_pack, inplace=False)
        single_student.eval()
        losses = _polish_block(single_student, bid, polish_tokens, POLISH_STEPS, POLISH_LR, teacher.device)
        if losses:
            print(f"    block {bid}: {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
        polished_weights[bid] = _extract_polished_weights(single_student, bid)
        del single_student

    print(f"  Building full compressed model...", flush=True)
    _set_seed(SEED)
    pack = _derive_block_pack_from_atlas(atlas, budget, FFN_BLOCKS)
    compressed_model = swap_compressed_ffns(teacher, pack, inplace=False)
    compressed_model.eval()
    for bid in FFN_BLOCKS:
        if bid in polished_weights and polished_weights[bid]:
            _inject_polished_weights(compressed_model, bid, polished_weights[bid])

    # Uncorrected PPL
    unc_metrics = _evaluate_model(teacher, compressed_model, corpus, batch_size=BATCH_SIZE)
    print(f"  Uncorrected PPL: {unc_metrics['perplexity']:.3f}", flush=True)

    # 3. Train P1 corrections
    print(f"\nTraining P1 corrections (r={RANK}, {CORRECTION_TRAIN_STEPS} steps)...", flush=True)
    p1_model = P1Model(compressed_model, FFN_BLOCKS, rank=RANK, d_model=D_MODEL)
    p1_model.install_hooks()

    train_losses = _train_corrections(p1_model, corpus.probe_holdout,
                                       CORRECTION_TRAIN_STEPS, CORRECTION_LR,
                                       teacher.device, SEED)
    p1_model.remove_hooks()

    # Eval corrected PPL
    p1_model.install_hooks()
    corr_metrics = _evaluate_model(teacher, compressed_model, corpus, batch_size=BATCH_SIZE)
    p1_model.remove_hooks()
    p1_ppl = corr_metrics["perplexity"]
    print(f"  Corrected P1 PPL: {p1_ppl:.3f} (cycle 125 ref: {CYCLE125_P1_SEED42})", flush=True)
    ppl_delta = abs(p1_ppl - CYCLE125_P1_SEED42)
    print(f"  PPL delta from reference: {ppl_delta:.3f}", flush=True)
    if ppl_delta > 2.0:
        print("  WARNING: PPL delta > 2, seed/config mismatch!", flush=True)

    # 4. Extract U, V matrices and compute SVD of C = U @ V
    print(f"\nExtracting correction matrices...", flush=True)
    correction_svds = {}
    for bid in ANALYSIS_BLOCKS:
        corr = p1_model.corrections[str(bid)]
        U = corr.U.data.clone()  # (768, r)
        V = corr.V.data.clone()  # (r, 768)
        C = U @ V  # (768, 768)
        U_c, S_c, Vt_c = svd_top_k(C, TOP_K_ALIGN)
        correction_svds[bid] = {
            "U_top": U_c,  # (768, k)
            "S_top": S_c,
            "C_fro": float(C.norm().item()),
        }
        print(f"  Block {bid}: ||C||_F={C.norm().item():.4f}, top-{TOP_K_ALIGN} singular values: {S_c.tolist()}", flush=True)

    # 5. Compression error SVD (post-FFN, pre-LayerNorm)
    print(f"\nComputing compression error E_l...", flush=True)
    holdout = corpus.eval_val
    float_acts = collect_residual_stream(float_model, holdout, ANALYSIS_BLOCKS, teacher.device)
    comp_acts = collect_residual_stream(compressed_model, holdout, ANALYSIS_BLOCKS, teacher.device)

    error_svds = {}
    for bid in ANALYSIS_BLOCKS:
        E = float_acts[bid] - comp_acts[bid]  # (tokens, 768)
        U_e, S_e, _ = svd_top_k(E.T, TOP_K_ALIGN)  # SVD of (768, tokens), top-k left SVs
        error_svds[bid] = {
            "U_top": U_e,  # (768, k)
            "E_fro": float(E.norm().item()),
        }
        print(f"  Block {bid}: ||E||_F={E.norm().item():.4f}, top-{TOP_K_ALIGN} σ: {S_e.tolist()}", flush=True)

    # 6. Attention amplification directions
    print(f"\nComputing attention output delta...", flush=True)
    attn_deltas = collect_attention_output_delta(float_model, compressed_model, holdout,
                                                  ANALYSIS_BLOCKS, teacher.device)
    attn_svds = {}
    for bid in ANALYSIS_BLOCKS:
        D_attn = attn_deltas[bid]  # (tokens, 768)
        U_a, S_a, _ = svd_top_k(D_attn.T, TOP_K_ALIGN)
        attn_svds[bid] = {
            "U_top": U_a,  # (768, k)
            "D_fro": float(D_attn.norm().item()),
        }
        print(f"  Block {bid}: ||D_attn||_F={D_attn.norm().item():.4f}, top-{TOP_K_ALIGN} σ: {S_a.tolist()}", flush=True)

    # 7. Bump channel directions
    print(f"\nComputing bump channel directions...", flush=True)
    bump_dirs = compute_bump_directions(float_model, holdout, ANALYSIS_BLOCKS, teacher.device)

    # 8. Null baseline
    print(f"\nComputing null baseline ({NULL_TRIALS} trials)...", flush=True)
    null_stats = null_baseline_max_cos(D_MODEL, TOP_K_ALIGN, NULL_TRIALS)
    print(f"  Null max|cos| 4x4 in R^768: mean={null_stats['mean']:.4f}, p99={null_stats['p99']:.4f}", flush=True)

    # 9. Compute alignment matrices
    print(f"\n{'='*60}", flush=True)
    print("ALIGNMENT RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    results_per_block = {}
    for bid in ANALYSIS_BLOCKS:
        corr_vecs = correction_svds[bid]["U_top"]  # (768, k)
        err_vecs = error_svds[bid]["U_top"]
        attn_vecs = attn_svds[bid]["U_top"]
        bump_vecs = bump_dirs[bid]  # (768, k) or fewer

        # Alignment matrices
        align_error = cosine_alignment_matrix(corr_vecs, err_vecs)  # (k, k)
        align_attn = cosine_alignment_matrix(corr_vecs, attn_vecs)
        align_bump = cosine_alignment_matrix(corr_vecs, bump_vecs)

        max_error = float(align_error.max())
        max_attn = float(align_attn.max())
        max_bump = float(align_bump.max())

        results_per_block[bid] = {
            "align_vs_error": align_error.tolist(),
            "align_vs_attn": align_attn.tolist(),
            "align_vs_bump": align_bump.tolist(),
            "max_cos_error": max_error,
            "max_cos_attn": max_attn,
            "max_cos_bump": max_bump,
            "correction_singular_values": correction_svds[bid]["S_top"].tolist(),
            "correction_fro": correction_svds[bid]["C_fro"],
            "error_fro": error_svds[bid]["E_fro"],
            "attn_delta_fro": attn_svds[bid]["D_fro"],
        }

        print(f"\n  Block {bid}:", flush=True)
        print(f"    max|cos| vs error:  {max_error:.4f}", flush=True)
        print(f"    max|cos| vs attn:   {max_attn:.4f}", flush=True)
        print(f"    max|cos| vs bump:   {max_bump:.4f}", flush=True)
        print(f"    null p99:           {null_stats['p99']:.4f}", flush=True)

    # 10. Verdict
    print(f"\n{'='*60}", flush=True)
    print("VERDICT", flush=True)
    print(f"{'='*60}", flush=True)

    # Count blocks where max alignment > 0.5 for each reference
    attn_above_05 = sum(1 for bid in ANALYSIS_BLOCKS if results_per_block[bid]["max_cos_attn"] > 0.5)
    error_above_05 = sum(1 for bid in ANALYSIS_BLOCKS if results_per_block[bid]["max_cos_error"] > 0.5)
    both_below_03 = sum(1 for bid in ANALYSIS_BLOCKS
                        if results_per_block[bid]["max_cos_attn"] < 0.3
                        and results_per_block[bid]["max_cos_error"] < 0.3)

    if attn_above_05 >= 3:
        verdict = "CONFIRMED"
        detail = f"Corrections align with attention amplification (cos>0.5 in {attn_above_05}/4 blocks)"
    elif error_above_05 >= 3:
        verdict = "REFUTED"
        detail = f"Corrections align with compression error (cos>0.5 in {error_above_05}/4 blocks)"
    elif both_below_03 >= 3:
        verdict = "INCONCLUSIVE"
        detail = f"Low alignment with both (<0.3 in {both_below_03}/4 blocks) — corrections learn novel subspace"
    else:
        verdict = "INCONCLUSIVE"
        detail = f"Mixed: attn>0.5 in {attn_above_05}/4, error>0.5 in {error_above_05}/4"

    print(f"  Attn alignment >0.5: {attn_above_05}/4 blocks", flush=True)
    print(f"  Error alignment >0.5: {error_above_05}/4 blocks", flush=True)
    print(f"  Both <0.3: {both_below_03}/4 blocks", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  Detail: {detail}", flush=True)

    # Save JSON
    output = {
        "cycle": 127,
        "item": 144,
        "seed": SEED,
        "float_ppl": bl_metrics["perplexity"],
        "uncorrected_ppl": unc_metrics["perplexity"],
        "corrected_ppl": p1_ppl,
        "ppl_delta_from_ref": ppl_delta,
        "null_baseline": null_stats,
        "per_block": {str(bid): results_per_block[bid] for bid in ANALYSIS_BLOCKS},
        "attn_above_05": attn_above_05,
        "error_above_05": error_above_05,
        "both_below_03": both_below_03,
        "verdict": verdict,
        "verdict_detail": detail,
    }
    json_path = RESULTS_DIR / "cycle_127_svd_alignment.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON: {json_path}", flush=True)

    # Plot: 4-block x 3-reference alignment heatmap
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, bid in enumerate(ANALYSIS_BLOCKS):
        ax = axes[i]
        data = results_per_block[bid]
        vals = [data["max_cos_error"], data["max_cos_attn"], data["max_cos_bump"], null_stats["p99"]]
        labels = ["Error", "Attn", "Bump", "Null p99"]
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#95a5a6"]

        ax.barh(range(len(labels)), vals, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="black", linestyle=":", alpha=0.5)
        ax.axvline(0.3, color="gray", linestyle="--", alpha=0.3)
        ax.set_title(f"Block {bid}")
        for j, v in enumerate(vals):
            ax.text(v + 0.02, j, f"{v:.3f}", va="center", fontsize=8)

    fig.suptitle(f"Correction SVD Alignment — Verdict: {verdict}", fontsize=12)
    plt.tight_layout()
    plot_path = RESULTS_DIR / "cycle_127_alignment_heatmap.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Saved plot: {plot_path}", flush=True)

    print("\nDone.", flush=True)
