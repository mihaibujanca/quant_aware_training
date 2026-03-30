"""Cycle 124: Attention entropy shift under FFN compression — GPT-2 (item 141).

Measures per-head attention entropy for float vs k=4 compressed model.
Correlates entropy shift with cycle 104 cascade amplification fractions.

Uses output_attentions=True to get post-softmax attention weights directly
from the model — no hooks on c_attn needed.

Verdict:
  CONFIRMED    if mean |delta_H| > 0.1 nats in >=8/12 layers AND
               |Spearman| > 0.3 AND p < 0.10 (correlation)
  REFUTED      if |delta_H| < 0.02 nats everywhere
  INCONCLUSIVE otherwise (shift exists but no significant correlation)
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
from scipy.stats import spearmanr

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
COMPRESS_BLOCKS = [1, 4, 7, 11]
HIDDEN_FRACTION = 1 / 6
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
EVAL_SEQS = 200
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
CYCLE104_PATH = ROOT / "autoresearch" / "results" / "cycle_104_cascade_source.json"

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
# Attention entropy computation via manual QKV + softmax
# (compatible with sdpa and compressed models that can't use output_attentions)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _compute_entropy_per_layer_head(model: nn.Module, tokens: torch.Tensor, device: torch.device):
    """
    Returns H[layer, head] = mean entropy over all (batch, query_pos) positions.

    Computes attention patterns manually: QKV from c_attn, causal softmax, then entropy.
    This works for both float and compressed models regardless of attn_implementation.
    The attention weights are the same as post-softmax weights (output_attentions=True would give).
    """
    H_accum = torch.zeros(N_LAYERS, N_HEADS, device="cpu")
    count = 0

    model.eval()
    n_batches = tokens.shape[0] // BATCH_SIZE
    blocks = model.transformer.h

    for bi in range(n_batches):
        batch = tokens[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE].to(device)
        inputs = batch[:, :-1].contiguous()  # [B, T-1]
        B, T = inputs.shape

        # Forward pass collecting per-layer QKV manually using hooks on ln_1 output
        # We hook the attention layer's c_attn output to get Q, K, V
        attn_weight_cache: dict[int, torch.Tensor] = {}

        hooks = []
        for layer_idx, block in enumerate(blocks):
            def make_hook(lidx):
                def hook(module, inp, out):
                    # out is the c_attn output: [B, T, 3*n_embd]
                    n_embd = N_HEADS * HEAD_DIM
                    q, k, v = out.split(n_embd, dim=-1)
                    # Split into heads: [B, N_HEADS, T, HEAD_DIM]
                    def sh(t):
                        return t.view(B, T, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
                    Q, K, V = sh(q), sh(k), sh(v)
                    scale = 1.0 / math.sqrt(HEAD_DIM)
                    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, N_HEADS, T, T]
                    causal = torch.triu(torch.ones(T, T, device=Q.device, dtype=torch.bool), diagonal=1)
                    scores = scores.masked_fill(causal[None, None], float('-inf'))
                    A = torch.softmax(scores, dim=-1)  # [B, N_HEADS, T, T]
                    A = torch.nan_to_num(A, nan=0.0)
                    attn_weight_cache[lidx] = A.cpu()
                return hook
            h = block.attn.c_attn.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)

        with torch.no_grad():
            model(inputs)

        for h in hooks:
            h.remove()

        # Compute entropy per layer per head
        for layer_idx in range(N_LAYERS):
            A = attn_weight_cache[layer_idx].to(device)  # [B, N_HEADS, T, T]
            eps = 1e-10
            ent = -(A * torch.log(A + eps)).sum(dim=-1)  # [B, N_HEADS, T]
            H_accum[layer_idx] += ent.mean(dim=(0, 2)).cpu()  # mean over B and T

        count += 1

    H = H_accum / max(count, 1)  # [N_LAYERS, N_HEADS]

    # Validate entropy range [0, ln(T)]
    T_val = tokens.shape[1] - 1  # seq len after truncation
    max_entropy = math.log(T_val)
    if H.max().item() > max_entropy + 0.01:
        print(f"  WARNING: max entropy {H.max().item():.3f} > ln({T_val})={max_entropy:.3f}")
    if H.min().item() < -0.01:
        print(f"  WARNING: min entropy {H.min().item():.3f} < 0")

    return H  # [N_LAYERS, N_HEADS]


# ---------------------------------------------------------------------------
# Load cycle 104 attn_frac values
# ---------------------------------------------------------------------------
def _load_cycle104_attn_frac():
    """Returns list of attn_frac values for transitions 0..10 (length 11)."""
    with open(CYCLE104_PATH) as f:
        d = json.load(f)
    abt = d["avg_by_transition"]
    fracs = [abt[str(i)]["attn_frac"] for i in range(11)]
    return fracs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _set_seed(42)

    # Load cycle 104 attn_frac
    attn_fracs = _load_cycle104_attn_frac()
    print(f"Cycle 104 attn_frac [transitions 0-10]: {[f'{v:.3f}' for v in attn_fracs]}")

    # Load teacher once
    print("Loading GPT-2 teacher...")
    teacher = load_nanochat_teacher(TeacherLoadSpec(device="auto"))
    global DEVICE
    DEVICE = teacher.device
    print(f"Device: {DEVICE}")

    # Build probe corpus
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

    # Tokenize eval tokens for entropy measurement: TOKENS_PER_SEED tokens = n_batches * BATCH_SIZE * SEQ_LEN
    n_seqs = TOKENS_PER_SEED // SEQ_LEN
    # Use probe_holdout for entropy measurement (already tokenized)
    holdout_tokens = corpus.probe_holdout  # [N, SEQ_LEN]
    print(f"Holdout tokens shape: {holdout_tokens.shape}, using first {n_seqs} seqs")
    # We'll use different offsets per seed

    all_seed_results = []

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n=== Seed {seed} ===")
        _set_seed(seed)

        # Different OWT batches per seed: offset by seed_idx * n_seqs
        seq_start = seed_idx * n_seqs
        seq_end = seq_start + n_seqs
        if seq_end > holdout_tokens.shape[0]:
            # Wrap around
            seq_end = holdout_tokens.shape[0]
            seq_start = max(0, seq_end - n_seqs)
        eval_tokens = holdout_tokens[seq_start:seq_end]  # [n_seqs, SEQ_LEN]
        print(f"  Using seqs [{seq_start}:{seq_end}] = {eval_tokens.shape[0]} seqs")

        # Build compressed model
        print(f"  Building k=4 compressed model...")
        compressed = _build_k4_compressed(teacher, corpus, seed)

        # PPL check
        float_model = teacher.model
        float_metrics = _evaluate_model(teacher, float_model, corpus, batch_size=BATCH_SIZE)
        comp_metrics = _evaluate_model(teacher, compressed, corpus, batch_size=BATCH_SIZE)
        float_ppl = float_metrics["perplexity"]
        comp_ppl = comp_metrics["perplexity"]
        print(f"  PPL: float={float_ppl:.2f}, compressed={comp_ppl:.2f}, ratio={comp_ppl/float_ppl:.2f}")

        if not (PPL_ABORT_MIN <= float_ppl <= PPL_ABORT_MAX):
            raise RuntimeError(f"Float PPL {float_ppl:.2f} outside [{PPL_ABORT_MIN}, {PPL_ABORT_MAX}]")
        if comp_ppl > 3.0 * float_ppl:
            raise RuntimeError(f"Compressed PPL {comp_ppl:.2f} > 3x float {float_ppl:.2f}")

        # Compute entropy for float model
        print(f"  Computing float entropy...")
        H_float = _compute_entropy_per_layer_head(float_model, eval_tokens, DEVICE)

        # Compute entropy for compressed model
        print(f"  Computing compressed entropy...")
        H_comp = _compute_entropy_per_layer_head(compressed, eval_tokens, DEVICE)

        delta_H = H_comp - H_float  # [N_LAYERS, N_HEADS], positive = more diffuse

        print(f"  delta_H stats: mean={delta_H.mean().item():.4f}, "
              f"abs_mean={delta_H.abs().mean().item():.4f}, "
              f"max_abs={delta_H.abs().max().item():.4f}")

        all_seed_results.append({
            "seed": seed,
            "float_ppl": float(float_ppl),
            "comp_ppl": float(comp_ppl),
            "H_float": H_float.tolist(),
            "H_comp": H_comp.tolist(),
            "delta_H": delta_H.tolist(),
        })

    # ---------------------------------------------------------------------------
    # Aggregate across seeds
    # ---------------------------------------------------------------------------
    print("\n=== Aggregating across seeds ===")
    delta_H_seeds = torch.stack([
        torch.tensor(r["delta_H"]) for r in all_seed_results
    ])  # [3, N_LAYERS, N_HEADS]

    mean_delta_H = delta_H_seeds.mean(dim=0)   # [N_LAYERS, N_HEADS]
    std_delta_H  = delta_H_seeds.std(dim=0)    # [N_LAYERS, N_HEADS]

    print(f"Cross-seed std of delta_H: mean={std_delta_H.mean().item():.4f}, "
          f"max={std_delta_H.max().item():.4f}")

    # Per-layer mean |delta_H| (averaged over heads)
    layer_mean_abs_dH = mean_delta_H.abs().mean(dim=1)  # [N_LAYERS]

    print("\nPer-layer mean |delta_H| (nats):")
    for L in range(N_LAYERS):
        marker = " <-- COMPRESSED" if L in COMPRESS_BLOCKS else ""
        print(f"  Layer {L:2d}: {layer_mean_abs_dH[L].item():.4f}{marker}")

    n_layers_above_threshold = (layer_mean_abs_dH > 0.1).sum().item()
    print(f"\nLayers with mean |delta_H| > 0.1 nats: {n_layers_above_threshold}/12")

    # ---------------------------------------------------------------------------
    # Spearman correlation with cycle 104 attn_frac
    # Map: transition i (i=0..10) -> layer i+1 (layers 1..11)
    # ---------------------------------------------------------------------------
    # Layer mean |delta_H| for layers 1-11
    layer_abs_dH_1_11 = layer_mean_abs_dH[1:12].numpy()  # [11]
    attn_frac_arr = np.array(attn_fracs)                  # [11]

    rho, p_val = spearmanr(layer_abs_dH_1_11, attn_frac_arr)
    print(f"\nSpearman: rho={rho:.4f}, p={p_val:.4f}")

    # ---------------------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------------------
    shift_exists = n_layers_above_threshold >= 8
    corr_significant = abs(rho) > 0.3 and p_val < 0.10

    if layer_mean_abs_dH.max().item() < 0.02:
        verdict = "REFUTED"
        verdict_note = "All |delta_H| < 0.02 nats — attention patterns stable under FFN compression"
    elif shift_exists and corr_significant:
        verdict = "CONFIRMED"
        verdict_note = (f"mean |delta_H| > 0.1 in {n_layers_above_threshold}/12 layers AND "
                        f"|Spearman|={abs(rho):.3f} > 0.3 with p={p_val:.4f} < 0.10")
    else:
        verdict = "INCONCLUSIVE"
        if shift_exists and not corr_significant:
            verdict_note = (f"Shift exists ({n_layers_above_threshold}/12 layers above 0.1 nats) "
                            f"but Spearman rho={rho:.3f} p={p_val:.4f} not significant")
        elif not shift_exists:
            verdict_note = (f"Shift below threshold: only {n_layers_above_threshold}/12 layers "
                            f"above 0.1 nats (max={layer_mean_abs_dH.max().item():.4f})")
        else:
            verdict_note = f"Spearman rho={rho:.3f} p={p_val:.4f}"

    print(f"\nVERDICT: {verdict}")
    print(f"  {verdict_note}")

    # ---------------------------------------------------------------------------
    # Save JSON results
    # ---------------------------------------------------------------------------
    results = {
        "verdict": verdict,
        "verdict_note": verdict_note,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "compress_blocks": COMPRESS_BLOCKS,
        "n_layers_above_0.1_nats": int(n_layers_above_threshold),
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "layer_mean_abs_dH": layer_mean_abs_dH.tolist(),
        "cycle104_attn_fracs": attn_fracs,
        "cross_seed_std_mean": float(std_delta_H.mean().item()),
        "cross_seed_std_max": float(std_delta_H.max().item()),
        "mean_delta_H": mean_delta_H.tolist(),   # [N_LAYERS, N_HEADS]
        "std_delta_H": std_delta_H.tolist(),     # [N_LAYERS, N_HEADS]
        "per_seed": all_seed_results,
    }

    json_path = RESULTS_DIR / "cycle_124_attn_entropy_shift.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # ---------------------------------------------------------------------------
    # 2-panel plot
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: per-layer mean |delta_H| bar chart
    ax1 = axes[0]
    layers = list(range(N_LAYERS))
    colors = ["#e74c3c" if L in COMPRESS_BLOCKS else "#3498db" for L in layers]
    ax1.bar(layers, layer_mean_abs_dH.numpy(), color=colors, alpha=0.8)
    ax1.axhline(0.1, color="k", linestyle="--", alpha=0.6, label="0.1 nat threshold")
    ax1.axhline(0.02, color="gray", linestyle=":", alpha=0.5, label="0.02 nat floor")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("|ΔH| (nats)")
    ax1.set_title("Per-layer mean |attention entropy shift|\n(red = compressed blocks)")
    ax1.set_xticks(layers)
    ax1.legend(fontsize=8)
    # Shade compressed blocks
    for cb in COMPRESS_BLOCKS:
        ax1.axvspan(cb - 0.4, cb + 0.4, alpha=0.05, color="red")

    # Panel 2: |delta_H| vs attn_frac scatter (layers 1-11)
    ax2 = axes[1]
    ax2.scatter(attn_frac_arr, layer_abs_dH_1_11, s=60, color="#2ecc71", alpha=0.8, zorder=3)
    for i, (x, y) in enumerate(zip(attn_frac_arr, layer_abs_dH_1_11)):
        layer_id = i + 1
        ax2.annotate(f"L{layer_id}", (x, y), textcoords="offset points", xytext=(5, 2), fontsize=8)
    # Fit line
    if len(attn_frac_arr) > 2:
        m, b = np.polyfit(attn_frac_arr, layer_abs_dH_1_11, 1)
        xs = np.linspace(attn_frac_arr.min(), attn_frac_arr.max(), 50)
        ax2.plot(xs, m * xs + b, "k--", alpha=0.5, linewidth=1)
    ax2.set_xlabel("Cycle 104 attn_frac")
    ax2.set_ylabel("|ΔH| (nats, mean over heads)")
    ax2.set_title(f"|ΔH| vs cascade attn_frac (layers 1-11)\n"
                  f"Spearman ρ={rho:.3f}, p={p_val:.3f}")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"GPT-2 k=4 FFN Compression — Attention Entropy Shift\n"
                 f"Verdict: {verdict} | {n_layers_above_threshold}/12 layers > 0.1 nats",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    png_path = RESULTS_DIR / "cycle_124_attn_entropy_shift.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {png_path}")

    return results


if __name__ == "__main__":
    os.makedirs(".mplconfig", exist_ok=True)
    main()
