"""Cycle 123: QK vs V error decomposition under FFN compression — GPT-2 (item 140).

Decomposes attention output error into:
  - QK-only error: hold V at float, use compressed attention patterns
  - V-only error:  hold attention pattern at float, use compressed V
  - Interaction:   E_total - E_QK - E_V

For k=4 compressed model (blocks [1,4,7,11] at 6x, topk+lstsq+500 Adam).
3 seeds [42, 123, 7], each with 2048 OWT tokens, 12 attention layers, 12 heads.

Verdict:
  CONFIRMED    if one component (QK or V) > 70% in >= 8/12 layers (same component, all 3 seeds)
  REFUTED      if all layers have both in [40-60%] across all seeds
  INCONCLUSIVE otherwise
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
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    _iter_lm_batches,
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
COMPRESS_BLOCKS = [1, 4, 7, 11]    # k=4 compressed blocks
HIDDEN_FRACTION = 1 / 6             # 6x compression
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
EVAL_SEQS = 200                     # for PPL check
SEQ_LEN = 128
BATCH_SIZE = 4
TOKENS_PER_SEED = 2048
N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
POLISH_STEPS = 500
POLISH_LR = 1e-4
PPL_ABORT_MIN = 20.0
PPL_ABORT_MAX = 80.0

OWT_PATH = ROOT / "data" / "openwebtext_sample.txt"
RESULTS_DIR = ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _find_mlp(model: nn.Module, block_id: int):
    try:
        return model.transformer.h[block_id].mlp
    except (AttributeError, IndexError):
        return None


def _polish_ffn(student: nn.Module, block_id: int, tokens: torch.Tensor, device: torch.device) -> None:
    """500-step Adam polish on W_down of block_id."""
    mlp = _find_mlp(student, block_id)
    if mlp is None or not hasattr(mlp, "c_proj"):
        return
    student.requires_grad_(False)
    w = mlp.c_proj.weight
    w.requires_grad_(True)
    b = getattr(mlp.c_proj, "bias", None)
    if b is not None:
        b.requires_grad_(True)
    params = [w] + ([b] if b is not None else [])
    opt = torch.optim.Adam(params, lr=POLISH_LR)
    student.train()
    n = tokens.shape[0]
    rng = torch.Generator()
    rng.manual_seed(42)
    for _ in range(POLISH_STEPS):
        idx = torch.randperm(n, generator=rng)[:BATCH_SIZE]
        batch = tokens[idx].to(device)
        inputs, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        opt.zero_grad()
        raw = student(inputs)
        logits = raw[0] if isinstance(raw, tuple) else (raw.logits if hasattr(raw, "logits") else raw)
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).backward()
        opt.step()
    student.eval()
    student.requires_grad_(False)


# ---------------------------------------------------------------------------
# Build k=4 compressed model
# ---------------------------------------------------------------------------
def _build_k4_compressed(teacher: TeacherBundle, corpus, seed: int) -> nn.Module:
    """Compress blocks [1,4,7,11] at 6x with topk+lstsq+500 Adam polish."""
    _set_seed(seed)
    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        selection_mode="topk",
    )
    atlas = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=tuple(COMPRESS_BLOCKS),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )
    pack = derive_ffn_compression(atlas, budget)
    student = swap_compressed_ffns(teacher, pack, inplace=False)
    device = teacher.device
    for block_id in COMPRESS_BLOCKS:
        _polish_ffn(student, block_id, corpus.probe_holdout, device)
    return student


# ---------------------------------------------------------------------------
# Attention QK/V decomposition — per head
# ---------------------------------------------------------------------------
@torch.no_grad()
def _decompose_attention_layer(
    float_block: nn.Module,
    comp_block: nn.Module,
    x_float: torch.Tensor,
    x_comp: torch.Tensor,
) -> dict:
    """
    Decompose attention output error per head for a single layer.

    Args:
        float_block: float model's transformer block
        comp_block:  compressed model's transformer block
        x_float: [B, T, D] residual stream entering this block in float model
        x_comp:  [B, T, D] residual stream entering this block in compressed model

    Returns:
        dict with per-head arrays (N_HEADS,) for:
          E_total, E_QK, E_V, E_inter, QK_frac, V_frac, inter_frac, KL
    """
    attn_float = float_block.attn
    attn_comp  = comp_block.attn

    n_embd = N_HEADS * HEAD_DIM  # 768
    B, T, _ = x_float.shape

    # Layer norm inputs
    h_float = float_block.ln_1(x_float)   # [B, T, 768]
    h_comp  = comp_block.ln_1(x_comp)     # [B, T, 768]

    # QKV projections
    def _qkv(attn, h):
        qkv = attn.c_attn(h)  # [B, T, 2304]
        q, k, v = qkv.split(n_embd, dim=-1)
        return q, k, v

    q_f, k_f, v_f = _qkv(attn_float, h_float)
    q_c, k_c, v_c = _qkv(attn_comp,  h_comp)

    # Split into heads: [B, N_HEADS, T, head_dim]
    def split_heads(t):
        return t.view(B, T, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

    Q_f = split_heads(q_f)  # [B, N_HEADS, T, HEAD_DIM]
    K_f = split_heads(k_f)
    V_f = split_heads(v_f)
    Q_c = split_heads(q_c)
    K_c = split_heads(k_c)
    V_c = split_heads(v_c)

    # Attention patterns (causal softmax)
    def _attn_pattern(Q, K):
        scale = 1.0 / math.sqrt(HEAD_DIM)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, N_HEADS, T, T]
        causal = torch.triu(torch.ones(T, T, device=Q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None], float('-inf'))
        A = torch.softmax(scores, dim=-1)
        A = torch.nan_to_num(A, nan=0.0)
        return A  # [B, N_HEADS, T, T]

    A_f = _attn_pattern(Q_f, K_f)  # float pattern
    A_c = _attn_pattern(Q_c, K_c)  # compressed pattern

    # c_proj weight: Conv1D layout [in_features=768, out_features=768]
    W_O_f = attn_float.c_proj.weight  # [768, 768]
    W_O_c = attn_comp.c_proj.weight

    # Per-head output projection function
    def _head_proj(ctx, W_O, head_idx):
        """ctx: [B, T, HEAD_DIM] -> [B, T, n_embd]"""
        W_h = W_O[head_idx * HEAD_DIM:(head_idx + 1) * HEAD_DIM, :]  # [HEAD_DIM, n_embd]
        return ctx @ W_h  # [B, T, n_embd]

    E_total_h = torch.zeros(N_HEADS, device=x_float.device)
    E_QK_h    = torch.zeros(N_HEADS, device=x_float.device)
    E_V_h     = torch.zeros(N_HEADS, device=x_float.device)
    KL_h      = torch.zeros(N_HEADS, device=x_float.device)

    for h in range(N_HEADS):
        # Float output for head h
        ctx_ff = torch.matmul(A_f[:, h], V_f[:, h])     # [B, T, HEAD_DIM]
        O_f_h  = _head_proj(ctx_ff, W_O_f, h)            # [B, T, n_embd]

        # Compressed output for head h
        ctx_cc = torch.matmul(A_c[:, h], V_c[:, h])
        O_c_h  = _head_proj(ctx_cc, W_O_c, h)

        # --- E_total: ||O_comp - O_float||^2 (mean over tokens) ---
        E_total_h[h] = (O_c_h - O_f_h).pow(2).sum(-1).mean()

        # --- E_QK: hold V at float, use compressed pattern ---
        ctx_qk = torch.matmul(A_c[:, h], V_f[:, h])     # compressed A, float V
        O_qk_h = _head_proj(ctx_qk, W_O_f, h)
        E_QK_h[h] = (O_qk_h - O_f_h).pow(2).sum(-1).mean()

        # --- E_V: hold pattern at float, use compressed V ---
        ctx_v  = torch.matmul(A_f[:, h], V_c[:, h])     # float A, compressed V
        O_v_h  = _head_proj(ctx_v, W_O_f, h)
        E_V_h[h] = (O_v_h - O_f_h).pow(2).sum(-1).mean()

        # --- KL divergence: KL(A_float || A_comp) per query position ---
        # A_f/A_c: [B, N_HEADS, T, T], select head h: [B, T, T]
        a_f_h = A_f[:, h]   # [B, T, T]
        a_c_h = A_c[:, h]
        # KL(p||q) = sum p * log(p / q), with clamping
        eps = 1e-10
        kl = (a_f_h * (torch.log(a_f_h.clamp(eps)) - torch.log(a_c_h.clamp(eps)))).sum(-1)  # [B, T]
        KL_h[h] = kl.mean()

    E_inter_h = E_total_h - E_QK_h - E_V_h

    # Fractions (clamp denominator to avoid div by zero)
    denom = E_total_h.clamp(min=1e-12)
    QK_frac_h   = (E_QK_h / denom).cpu()
    V_frac_h    = (E_V_h  / denom).cpu()
    inter_frac_h = (E_inter_h / denom).cpu()

    return {
        "E_total":    E_total_h.cpu().tolist(),
        "E_QK":       E_QK_h.cpu().tolist(),
        "E_V":        E_V_h.cpu().tolist(),
        "E_inter":    E_inter_h.cpu().tolist(),
        "QK_frac":    QK_frac_h.tolist(),
        "V_frac":     V_frac_h.tolist(),
        "inter_frac": inter_frac_h.tolist(),
        "KL":         KL_h.cpu().tolist(),
    }


# ---------------------------------------------------------------------------
# Run one seed
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_seed(
    float_model: nn.Module,
    comp_model: nn.Module,
    token_batch: torch.Tensor,   # [N, SEQ_LEN]
) -> list[dict]:
    """Run all 12 layers for a single token batch. Returns list[layer_dict]."""
    # We need residual stream at each layer for BOTH models simultaneously.
    # We'll do a synchronized layer-by-layer forward.
    B, T = token_batch.shape

    # Embeddings
    pos = torch.arange(T, device=DEVICE)
    x_f = float_model.transformer.drop(
        float_model.transformer.wte(token_batch) + float_model.transformer.wpe(pos)
    )
    x_c = comp_model.transformer.drop(
        comp_model.transformer.wte(token_batch) + comp_model.transformer.wpe(pos)
    )

    results = []
    for layer_idx in range(N_LAYERS):
        fb = float_model.transformer.h[layer_idx]
        cb = comp_model.transformer.h[layer_idx]

        # Decompose attention at this layer (using residual stream entering the block)
        layer_res = _decompose_attention_layer(fb, cb, x_f, x_c)
        results.append(layer_res)

        # Advance residual stream through this block for both models
        x_f = fb(x_f)[0]
        x_c = cb(x_c)[0]

    return results


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------
def compute_verdict(seed_results: list[list[list[dict]]]) -> dict:
    """
    seed_results: [seed][batch][layer] -> {QK_frac: [N_HEADS], V_frac: [N_HEADS], ...}
    Returns layer-mean fracs per seed, cross-seed mean/std, verdict.
    """
    # Per seed, per layer: mean across heads and batches
    # seed_results[s] = list of batch results, each batch is list[N_LAYERS] of layer_dicts
    n_seeds = len(seed_results)

    # Aggregate: for each seed, gather all batches, then mean over heads
    # seed_layer_means[s][l] = {'QK_frac_mean', 'V_frac_mean', 'inter_frac_mean', 'KL_mean'}
    seed_layer_means = []
    for s_batches in seed_results:
        # s_batches: list[batch] of list[layer] of dicts
        layer_data = {l: {"QK_frac": [], "V_frac": [], "inter_frac": [], "KL": []} for l in range(N_LAYERS)}
        for batch_layers in s_batches:
            for l, ldict in enumerate(batch_layers):
                layer_data[l]["QK_frac"].extend(ldict["QK_frac"])
                layer_data[l]["V_frac"].extend(ldict["V_frac"])
                layer_data[l]["inter_frac"].extend(ldict["inter_frac"])
                layer_data[l]["KL"].extend(ldict["KL"])
        layer_means = []
        for l in range(N_LAYERS):
            layer_means.append({
                "QK_frac_mean":   float(np.mean(layer_data[l]["QK_frac"])),
                "V_frac_mean":    float(np.mean(layer_data[l]["V_frac"])),
                "inter_frac_mean": float(np.mean(layer_data[l]["inter_frac"])),
                "KL_mean":        float(np.mean(layer_data[l]["KL"])),
            })
        seed_layer_means.append(layer_means)

    # Cross-seed mean/std per layer
    cross_seed = []
    for l in range(N_LAYERS):
        qk_vals  = [seed_layer_means[s][l]["QK_frac_mean"]   for s in range(n_seeds)]
        v_vals   = [seed_layer_means[s][l]["V_frac_mean"]    for s in range(n_seeds)]
        inter_v  = [seed_layer_means[s][l]["inter_frac_mean"] for s in range(n_seeds)]
        kl_vals  = [seed_layer_means[s][l]["KL_mean"]         for s in range(n_seeds)]
        cross_seed.append({
            "QK_frac_mean":    float(np.mean(qk_vals)),
            "QK_frac_std":     float(np.std(qk_vals)),
            "V_frac_mean":     float(np.mean(v_vals)),
            "V_frac_std":      float(np.std(v_vals)),
            "inter_frac_mean": float(np.mean(inter_v)),
            "inter_frac_std":  float(np.std(inter_v)),
            "KL_mean":         float(np.mean(kl_vals)),
            "KL_std":          float(np.std(kl_vals)),
        })

    # Verdict: count layers where all 3 seeds agree that QK_frac or V_frac > 0.70
    confirmed_qk = 0
    confirmed_v  = 0
    refuted_count = 0
    for l in range(N_LAYERS):
        qk_all = [seed_layer_means[s][l]["QK_frac_mean"] > 0.70 for s in range(n_seeds)]
        v_all  = [seed_layer_means[s][l]["V_frac_mean"]  > 0.70 for s in range(n_seeds)]
        if all(qk_all):
            confirmed_qk += 1
        if all(v_all):
            confirmed_v  += 1
        # Check refute condition: both in [0.40, 0.60] across all seeds
        both_equal = all(
            0.40 <= seed_layer_means[s][l]["QK_frac_mean"] <= 0.60 and
            0.40 <= seed_layer_means[s][l]["V_frac_mean"]  <= 0.60
            for s in range(n_seeds)
        )
        if both_equal:
            refuted_count += 1

    # Stability check
    max_std = max(cross_seed[l]["QK_frac_std"] for l in range(N_LAYERS))
    high_interaction = sum(
        1 for l in range(N_LAYERS)
        if abs(cross_seed[l]["inter_frac_mean"]) > 0.30
    )

    dominant_count = max(confirmed_qk, confirmed_v)
    dominant_component = "QK" if confirmed_qk >= confirmed_v else "V"
    if confirmed_qk == confirmed_v == 0:
        dominant_component = "neither"

    if dominant_count >= 8:
        verdict = "CONFIRMED"
        verdict_detail = f"{dominant_component} dominates in {dominant_count}/12 layers (all seeds)"
    elif refuted_count >= 10:
        verdict = "REFUTED"
        verdict_detail = f"QK/V roughly equal in {refuted_count}/12 layers"
    else:
        verdict = "INCONCLUSIVE"
        verdict_detail = f"QK dominant in {confirmed_qk}/12 layers, V dominant in {confirmed_v}/12 layers"

    warnings = []
    if max_std > 0.15:
        warnings.append(f"High cross-seed instability: max_std={max_std:.3f}")
    if high_interaction > 0:
        warnings.append(f"Interaction term > 30% in {high_interaction}/12 layers — additive decomposition is weak")

    return {
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "confirmed_qk_layers": confirmed_qk,
        "confirmed_v_layers": confirmed_v,
        "refuted_layers": refuted_count,
        "dominant_component": dominant_component,
        "dominant_count": dominant_count,
        "cross_seed_layer_stats": cross_seed,
        "seed_layer_means": seed_layer_means,
        "max_cross_seed_std": float(max_std),
        "high_interaction_layers": high_interaction,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(verdict_data: dict, per_head_per_seed: list, results_dir: Path) -> None:
    """
    Three-panel figure:
      (a) 12×12 QK_frac heatmap (layer × head, seed-averaged)
      (b) 12×12 KL heatmap (layer × head, seed-averaged)
      (c) Per-layer stacked bar: QK_frac / V_frac / inter_frac
    """
    cross_seed = verdict_data["cross_seed_layer_stats"]
    seed_layer_means = verdict_data["seed_layer_means"]

    n_seeds = len(per_head_per_seed)

    # Compute seed-averaged per-head heatmaps
    # per_head_per_seed[s] = list[batch] of list[layer] of dicts
    qk_heatmap = np.zeros((N_LAYERS, N_HEADS))
    kl_heatmap = np.zeros((N_LAYERS, N_HEADS))
    v_heatmap  = np.zeros((N_LAYERS, N_HEADS))
    count_mat  = np.zeros((N_LAYERS, N_HEADS))

    for s_batches in per_head_per_seed:
        for batch_layers in s_batches:
            for l, ldict in enumerate(batch_layers):
                for h in range(N_HEADS):
                    qk_heatmap[l, h] += ldict["QK_frac"][h]
                    v_heatmap[l, h]  += ldict["V_frac"][h]
                    kl_heatmap[l, h] += ldict["KL"][h]
                    count_mat[l, h]  += 1

    qk_heatmap /= count_mat.clip(min=1)
    v_heatmap  /= count_mat.clip(min=1)
    kl_heatmap /= count_mat.clip(min=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) QK_frac heatmap
    im0 = axes[0].imshow(qk_heatmap, vmin=0, vmax=1, aspect="auto", cmap="RdYlGn")
    axes[0].set_title("QK_frac (layer × head)", fontsize=11)
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    axes[0].set_xticks(range(N_HEADS))
    axes[0].set_yticks(range(N_LAYERS))
    plt.colorbar(im0, ax=axes[0])

    # (b) KL heatmap
    kl_vmax = np.percentile(kl_heatmap, 95)
    im1 = axes[1].imshow(kl_heatmap, vmin=0, vmax=max(kl_vmax, 1e-6), aspect="auto", cmap="hot_r")
    axes[1].set_title("KL(A_float || A_comp) (layer × head)", fontsize=11)
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    axes[1].set_xticks(range(N_HEADS))
    axes[1].set_yticks(range(N_LAYERS))
    plt.colorbar(im1, ax=axes[1])

    # (c) Stacked bar per layer
    layers = list(range(N_LAYERS))
    qk_vals   = [cross_seed[l]["QK_frac_mean"] for l in layers]
    v_vals    = [cross_seed[l]["V_frac_mean"]  for l in layers]
    inter_vals = [cross_seed[l]["inter_frac_mean"] for l in layers]

    bar_width = 0.6
    axes[2].bar(layers, qk_vals,    bar_width, label="QK_frac",    color="steelblue")
    axes[2].bar(layers, v_vals,     bar_width, bottom=qk_vals,     label="V_frac",    color="coral")
    inter_bottom = [qk_vals[l] + v_vals[l] for l in layers]
    axes[2].bar(layers, inter_vals, bar_width, bottom=inter_bottom, label="Inter_frac", color="gray", alpha=0.6)
    axes[2].axhline(0.70, color="black", linestyle="--", linewidth=1.0, label="70% threshold")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Fraction")
    axes[2].set_title("Error decomposition per layer (seed-avg)", fontsize=11)
    axes[2].set_xticks(layers)
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(-0.4, 1.4)

    plt.suptitle(
        f"GPT-2 k=4 FFN Compression — QK vs V Error Decomposition\n"
        f"Verdict: {verdict_data['verdict']} — {verdict_data['verdict_detail']}",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = results_dir / "cycle_123_qk_v_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global DEVICE

    # --- Load teacher ---
    print("Loading GPT-2 teacher model...")
    teacher = load_nanochat_teacher(TeacherLoadSpec(device="auto"))
    DEVICE = teacher.device
    print(f"Device: {DEVICE}")
    float_model = teacher.model
    float_model.eval()

    # --- Build probe corpus (used to compute compression) ---
    corpus = build_probe_corpus(
        ProbeCorpusSpec(
            text_path=str(OWT_PATH),
            sequence_length=SEQ_LEN,
            probe_train_sequences=PROBE_TRAIN_SEQS,
            probe_holdout_sequences=PROBE_TRAIN_SEQS,
            eval_val_sequences=EVAL_SEQS,
            teacher=teacher,
        )
    )

    # --- Evaluate float model PPL ---
    float_metrics = _evaluate_model(teacher, float_model, corpus)
    float_ppl = float_metrics["perplexity"]
    print(f"Float model PPL: {float_ppl:.2f}")

    # --- Build k=4 compressed model (seed 42 for compression; seeds vary OWT batch only) ---
    print("Building k=4 compressed model (blocks [1,4,7,11] at 6x)...")
    comp_model = _build_k4_compressed(teacher, corpus, seed=42)
    comp_model.eval()

    # --- Evaluate compressed model PPL ---
    comp_metrics = _evaluate_model(teacher, comp_model, corpus)
    comp_ppl = comp_metrics["perplexity"]
    ppl_change_pct = 100.0 * (comp_ppl - float_ppl) / float_ppl
    print(f"Compressed model PPL: {comp_ppl:.2f} ({ppl_change_pct:+.1f}%)")
    if comp_ppl < PPL_ABORT_MIN or comp_ppl > PPL_ABORT_MAX:
        print(f"WARNING: Compressed PPL {comp_ppl:.2f} outside expected range [{PPL_ABORT_MIN}, {PPL_ABORT_MAX}]")

    # --- Tokenize OWT for per-seed batches ---
    # We'll use the OWT text to produce 3 × TOKENS_PER_SEED tokens with different seeds
    print("Tokenizing OWT for seed batches...")
    all_token_ids = []
    with open(OWT_PATH, "r") as f:
        text = f.read()
    tokens = teacher.tokenizer.encode(text)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    print(f"Total OWT tokens available: {len(tokens_tensor)}")

    # --- Run per-seed experiment ---
    all_seed_results = []       # [seed][batch_idx][layer] -> dict
    all_ppl = []

    for seed_idx, seed in enumerate(SEEDS):
        _set_seed(seed)
        print(f"\n=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===")

        # Sample TOKENS_PER_SEED contiguous tokens from a random offset
        max_start = len(tokens_tensor) - TOKENS_PER_SEED - SEQ_LEN
        start = torch.randint(0, max(1, max_start), (1,)).item()
        seed_tokens = tokens_tensor[start:start + TOKENS_PER_SEED]

        # Split into batches of SEQ_LEN
        # We'll process as one batch of shape [TOKENS_PER_SEED // SEQ_LEN, SEQ_LEN]
        n_seqs = TOKENS_PER_SEED // SEQ_LEN
        seed_batch = seed_tokens[:n_seqs * SEQ_LEN].view(n_seqs, SEQ_LEN).to(DEVICE)
        print(f"  Batch shape: {seed_batch.shape}")

        # Process in smaller chunks to avoid OOM on MPS
        chunk_size = 4
        seed_layer_results = []   # list[batch_chunk] -> list[layer] of dicts

        for chunk_start in range(0, n_seqs, chunk_size):
            chunk = seed_batch[chunk_start:chunk_start + chunk_size]
            layer_results = run_seed(float_model, comp_model, chunk)
            seed_layer_results.append(layer_results)
            if chunk_start == 0:
                # Sanity check layer 0: total error should be ~0 (no upstream compression)
                e0 = sum(layer_results[0]["E_total"])
                print(f"  Layer 0 E_total (should be ~0 if no earlier compression): {e0:.4f}")
                e0_qk_sum = sum(layer_results[0]["QK_frac"]) / N_HEADS
                e0_v_sum  = sum(layer_results[0]["V_frac"])  / N_HEADS
                print(f"  Layer 0 mean QK_frac={e0_qk_sum:.3f}, V_frac={e0_v_sum:.3f}")

        all_seed_results.append(seed_layer_results)

        # Sample per-layer fracs for this seed
        for l in range(N_LAYERS):
            qk_vals = []
            v_vals  = []
            for batch_layers in seed_layer_results:
                qk_vals.extend(batch_layers[l]["QK_frac"])
                v_vals.extend(batch_layers[l]["V_frac"])
            print(f"  Layer {l:2d}: QK_frac={np.mean(qk_vals):.3f} V_frac={np.mean(v_vals):.3f}")

    # --- Compute verdict ---
    print("\nComputing verdict...")
    verdict_data = compute_verdict(all_seed_results)
    print(f"Verdict: {verdict_data['verdict']}")
    print(f"  {verdict_data['verdict_detail']}")
    if verdict_data["warnings"]:
        for w in verdict_data["warnings"]:
            print(f"  WARNING: {w}")

    # --- Sanity checks ---
    print("\nSanity checks:")
    # 1) E_QK + E_V + E_inter ≈ E_total (check relative error < 1e-4)
    for s_idx, s_batches in enumerate(all_seed_results):
        for b_idx, batch_layers in enumerate(s_batches):
            for l, ldict in enumerate(batch_layers):
                for h in range(N_HEADS):
                    tot  = ldict["E_total"][h]
                    recon = ldict["E_QK"][h] + ldict["E_V"][h] + ldict["E_inter"][h]
                    rel_err = abs(tot - recon) / (tot + 1e-12)
                    if rel_err > 1e-3:
                        print(f"  FAIL decomp sanity s={s_idx} b={b_idx} l={l} h={h}: rel_err={rel_err:.2e}")

    # 2) KL values non-negative
    kl_neg = 0
    kl_nan = 0
    for s_batches in all_seed_results:
        for batch_layers in s_batches:
            for ldict in batch_layers:
                for kl_val in ldict["KL"]:
                    if kl_val < -1e-6:
                        kl_neg += 1
                    if not math.isfinite(kl_val):
                        kl_nan += 1
    print(f"  KL negative count: {kl_neg}, NaN/Inf count: {kl_nan}")

    # 3) Cross-seed std
    max_std = verdict_data["max_cross_seed_std"]
    print(f"  Max cross-seed std: {max_std:.3f} ({'OK' if max_std < 0.15 else 'HIGH'})")

    # --- Plots ---
    print("\nGenerating plots...")
    make_plots(verdict_data, all_seed_results, RESULTS_DIR)

    # --- Save JSON results ---
    results = {
        "cycle": 123,
        "item": 140,
        "compress_blocks": COMPRESS_BLOCKS,
        "hidden_fraction": HIDDEN_FRACTION,
        "seeds": SEEDS,
        "tokens_per_seed": TOKENS_PER_SEED,
        "float_ppl": float(float_ppl),
        "comp_ppl": float(comp_ppl),
        "ppl_change_pct": float(ppl_change_pct),
        "verdict": verdict_data["verdict"],
        "verdict_detail": verdict_data["verdict_detail"],
        "confirmed_qk_layers": verdict_data["confirmed_qk_layers"],
        "confirmed_v_layers": verdict_data["confirmed_v_layers"],
        "refuted_layers": verdict_data["refuted_layers"],
        "dominant_component": verdict_data["dominant_component"],
        "dominant_count": verdict_data["dominant_count"],
        "max_cross_seed_std": verdict_data["max_cross_seed_std"],
        "high_interaction_layers": verdict_data["high_interaction_layers"],
        "warnings": verdict_data["warnings"],
        "cross_seed_layer_stats": verdict_data["cross_seed_layer_stats"],
        "seed_layer_means": verdict_data["seed_layer_means"],
        "sanity": {
            "kl_negative_count": kl_neg,
            "kl_nan_count": kl_nan,
        },
    }

    json_path = RESULTS_DIR / "cycle_123_qk_v_decomposition.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {json_path}")

    # Summary
    print("\n" + "="*60)
    print(f"VERDICT: {verdict_data['verdict']}")
    print(f"  {verdict_data['verdict_detail']}")
    print(f"  Float PPL: {float_ppl:.2f}, Compressed PPL: {comp_ppl:.2f} ({ppl_change_pct:+.1f}%)")
    print(f"  QK dominant layers: {verdict_data['confirmed_qk_layers']}/12")
    print(f"  V dominant layers:  {verdict_data['confirmed_v_layers']}/12")
    cs = verdict_data["cross_seed_layer_stats"]
    print("\n  Cross-seed layer summary:")
    for l in range(N_LAYERS):
        marker = " [COMPRESSED FFN]" if l in COMPRESS_BLOCKS else ""
        print(f"  L{l:2d}: QK={cs[l]['QK_frac_mean']:.3f}±{cs[l]['QK_frac_std']:.3f} "
              f"V={cs[l]['V_frac_mean']:.3f}±{cs[l]['V_frac_std']:.3f} "
              f"inter={cs[l]['inter_frac_mean']:.3f}±{cs[l]['inter_frac_std']:.3f} "
              f"KL={cs[l]['KL_mean']:.4f}{marker}")
    print("="*60)


if __name__ == "__main__":
    main()
