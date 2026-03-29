"""Cycle 113: Per-head amplification census — GPT-2 (item 124).

For each of 11 transitions (compress FFN block L, measure attention in block L+1):
1. Compress FFN block L at 6x using topk+lstsq+500 Adam.
2. Hook per-head outputs in block L+1's attention layer.
3. Run 256 OWT seqs through float and compressed models.
4. Per head h: amp_h = mean_tokens(||head_out_perturbed - head_out_float||_2) / mean_tokens(||delta_L||_2)
5. Compute top-3 share and max share per layer.

Verdict:
- Confirmed if top-3 heads >= 70% of amplification in >= 8/11 layers.
- Refuted if max per head <= 15% in >= 8/11 layers.
- Inconclusive otherwise.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import List

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
    FfnGeometryAtlas,
    ProbeCorpusSpec,
    TeacherBundle,
    TeacherLoadSpec,
    _evaluate_model,
    _fit_output_projection,
    _iter_lm_batches,
    _layer_target_hidden,
    _select_hidden_units,
    _standard_bias,
    _standard_weight,
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
EVAL_SEQS = 256
SEQ_LEN = 128
BATCH_SIZE = 4
HIDDEN_FRACTION = 1 / 6         # 6x compression
MAX_TOKENS_PER_LAYER = 4096
PROBE_TRAIN_SEQS = 200
N_TRANSITIONS = 11              # blocks 0..10 → 1..11
N_HEADS = 12                    # GPT-2: 12 heads, each dim 64
HEAD_DIM = 64
PPL_ABORT_MIN = 25.0
PPL_ABORT_MAX = 65.0
POLISH_STEPS = 500
POLISH_LR = 1e-4

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


def _tok_norm(x: torch.Tensor) -> torch.Tensor:
    """Mean L2 norm over all tokens in batch. Returns scalar."""
    return x.norm(dim=-1).mean()


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
# Compress single block FFN
# ---------------------------------------------------------------------------
def _compress_single_block(teacher: TeacherBundle, corpus, block_id: int, seed: int) -> nn.Module:
    """Return a student model with only block `block_id` FFN compressed (6x topk+lstsq+500 Adam)."""
    _set_seed(seed)
    budget = CompressionBudgetSpec(
        hidden_fraction=HIDDEN_FRACTION,
        selection_mode="topk",
    )
    atlas = probe_ffn_geometry(
        teacher,
        corpus,
        layer_ids=(block_id,),
        max_tokens_per_layer=MAX_TOKENS_PER_LAYER,
        batch_size=BATCH_SIZE,
    )
    pack = derive_ffn_compression(atlas, budget)
    student = swap_compressed_ffns(teacher, pack, inplace=False)
    _polish_ffn(student, block_id, corpus.probe_holdout, DEVICE)
    return student


# ---------------------------------------------------------------------------
# Per-head attention output extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def _compute_per_head_outputs(
    block: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-head attention output contributions for GPT-2 block.

    GPT-2 Conv1D layout: weight is [in_features, out_features].
    c_attn: [768, 2304], c_proj: [768, 768].

    For each head h:
      q_h, k_h, v_h: [B, T, head_dim]
      ctx_h = softmax(q_h @ k_h^T / sqrt(d)) @ v_h  -> [B, T, head_dim]  (causal)
      proj_h = ctx_h @ W_O_h  (where W_O_h = c_proj.weight[h*d:(h+1)*d, :]) -> [B, T, n_embd]

    Returns: [N_HEADS, B*T, n_embd] per-head projected contributions.
    """
    attn = block.attn
    # Conv1D weight shape: [in_features, out_features]
    n_embd = attn.c_proj.weight.shape[1]  # out_features = 768

    h_ln = block.ln_1(x)
    B, T, _ = h_ln.shape

    # c_attn: [B, T, 768] -> [B, T, 2304]
    qkv = attn.c_attn(h_ln)
    q, k, v = qkv.split(n_embd, dim=-1)  # each [B, T, n_embd]

    # Split into heads: [B, T, N_HEADS, head_dim] -> [B, N_HEADS, T, head_dim]
    def split_heads(t):
        return t.view(B, T, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

    q_h = split_heads(q)  # [B, N_HEADS, T, head_dim]
    k_h = split_heads(k)
    v_h = split_heads(v)

    # Attention scores with causal mask
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale  # [B, N_HEADS, T, T]

    # Apply causal mask: positions j > i get -inf
    causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    weights = torch.softmax(scores, dim=-1)  # [B, N_HEADS, T, T]
    # Replace NaN from all-inf rows (shouldn't happen for T>0 with causal)
    weights = torch.nan_to_num(weights, nan=0.0)

    # Context vectors: [B, N_HEADS, T, head_dim]
    ctx = torch.matmul(weights, v_h)

    # Project each head through its slice of c_proj
    # c_proj.weight: [768, 768] (Conv1D: [in, out])
    # Head h occupies input rows [h*head_dim : (h+1)*head_dim]
    W_O = attn.c_proj.weight  # [n_embd, n_embd]

    head_contributions = []
    for i in range(N_HEADS):
        ctx_i = ctx[:, i, :, :]                       # [B, T, head_dim]
        W_O_i = W_O[i * HEAD_DIM:(i + 1) * HEAD_DIM, :]  # [head_dim, n_embd]
        proj_i = ctx_i @ W_O_i                        # [B, T, n_embd]
        head_contributions.append(proj_i.reshape(B * T, n_embd))

    return torch.stack(head_contributions, dim=0)  # [N_HEADS, B*T, n_embd]


# ---------------------------------------------------------------------------
# Run one transition
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_transition_per_head(
    float_model: nn.Module,
    student_model: nn.Module,
    corpus,
    source_block: int,
) -> dict:
    """Measure per-head amplification for transition source_block → source_block+1."""
    target_block = source_block + 1
    target_blk = float_model.transformer.h[target_block]

    # Accumulators: per head sum of norms and delta_L norms
    head_amp_sums = torch.zeros(N_HEADS, device=DEVICE)
    delta_L_norm_sum = 0.0
    n_batches = 0

    for ids, _ in _iter_lm_batches(corpus.eval_val, batch_size=BATCH_SIZE, device=DEVICE):
        input_ids = ids[:, :-1]
        B, T = input_ids.shape

        # Forward through shared prefix
        x = float_model.transformer.wte(input_ids) + float_model.transformer.wpe(
            torch.arange(T, device=DEVICE)
        )
        x = float_model.transformer.drop(x)

        for blk_idx in range(source_block):
            x = float_model.transformer.h[blk_idx](x)[0]

        x_pre_src = x

        # Float and compressed through source block
        x_float_after_src = float_model.transformer.h[source_block](x_pre_src)[0]
        x_comp_after_src = student_model.transformer.h[source_block](x_pre_src)[0]

        delta_L = x_comp_after_src - x_float_after_src
        delta_L_norm_sum += delta_L.norm(dim=-1).mean().item()

        # Per-head outputs for float and perturbed inputs
        head_float = _compute_per_head_outputs(target_blk, x_float_after_src)    # [N_HEADS, B*T, n_embd]
        head_perturbed = _compute_per_head_outputs(target_blk, x_comp_after_src)  # [N_HEADS, B*T, n_embd]

        diff = head_perturbed - head_float  # [N_HEADS, B*T, n_embd]
        per_head_norms = diff.norm(dim=-1).mean(dim=-1)  # [N_HEADS]
        head_amp_sums += per_head_norms
        n_batches += 1

    # Average
    head_amps = (head_amp_sums / n_batches).cpu()  # [N_HEADS]
    delta_L_mean = delta_L_norm_sum / n_batches

    # Normalize by delta_L norm to get amplification factors
    amp_factors = head_amps / (delta_L_mean + 1e-10)

    total_amp = amp_factors.sum().item()
    if total_amp > 1e-10:
        amp_fracs = (amp_factors / total_amp).tolist()
    else:
        amp_fracs = [1.0 / N_HEADS] * N_HEADS

    sorted_fracs = sorted(amp_fracs, reverse=True)
    top3_frac = sum(sorted_fracs[:3])
    max_frac = sorted_fracs[0]

    # Gini coefficient
    x_sorted = sorted(amp_fracs)
    n = len(x_sorted)
    cumulative = 0.0
    for i, v in enumerate(x_sorted):
        cumulative += (2 * (i + 1) - n - 1) * v
    gini = cumulative / (n * sum(x_sorted)) if sum(x_sorted) > 1e-10 else 0.0

    return {
        "source_block": source_block,
        "target_block": target_block,
        "amp_factors": amp_factors.tolist(),
        "amp_fracs": amp_fracs,
        "top3_frac": top3_frac,
        "max_frac": max_frac,
        "gini": gini,
        "delta_L_mean": delta_L_mean,
    }


# ---------------------------------------------------------------------------
# Ablation: head importance via PPL delta
# ---------------------------------------------------------------------------
@torch.no_grad()
def _ablation_head_importance(float_model: nn.Module, teacher: TeacherBundle, corpus, block_id: int) -> List[float]:
    """Zero out each head's c_proj contribution for block_id, return PPL deltas."""
    n_embd = float_model.transformer.h[block_id].attn.c_proj.weight.shape[1]

    ppl_deltas = []
    for head_idx in range(N_HEADS):
        ablated = copy.deepcopy(float_model)
        attn = ablated.transformer.h[block_id].attn
        # Zero out the rows corresponding to this head
        attn.c_proj.weight[head_idx * HEAD_DIM:(head_idx + 1) * HEAD_DIM, :] = 0.0
        if attn.c_proj.bias is not None:
            # Bias is shared — can't cleanly zero per head; skip bias ablation
            pass
        ablated.eval()
        metrics = _evaluate_model(teacher, ablated, corpus, batch_size=BATCH_SIZE)
        ppl_deltas.append(metrics["perplexity"])

    base_metrics = _evaluate_model(teacher, float_model, corpus, batch_size=BATCH_SIZE)
    base_ppl = base_metrics["perplexity"]
    return [p - base_ppl for p in ppl_deltas]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    _set_seed(SEED)

    print("Loading teacher...")
    load_spec = TeacherLoadSpec(
        source="hf_pretrained",
        hf_model_name="gpt2",
        device=str(DEVICE),
    )
    teacher = load_nanochat_teacher(load_spec)
    float_model = teacher.model
    float_model.eval()

    print("Building probe corpus...")
    probe_spec = ProbeCorpusSpec(
        probe_train_sequences=PROBE_TRAIN_SEQS,
        eval_val_sequences=EVAL_SEQS,
        sequence_length=SEQ_LEN,
        text_path=str(OWT_PATH),
        teacher=teacher,
    )
    corpus = build_probe_corpus(probe_spec)

    # Abort check
    base_metrics = _evaluate_model(teacher, float_model, corpus, batch_size=BATCH_SIZE)
    base_ppl = base_metrics["perplexity"]
    print(f"Baseline PPL: {base_ppl:.2f}")
    if not (PPL_ABORT_MIN <= base_ppl <= PPL_ABORT_MAX):
        print(f"ABORT: baseline PPL {base_ppl:.2f} outside [{PPL_ABORT_MIN}, {PPL_ABORT_MAX}]")
        sys.exit(1)

    all_transitions = []
    heatmap_data = np.zeros((N_TRANSITIONS, N_HEADS))  # rows=transition (src block), cols=head

    for src_block in range(N_TRANSITIONS):
        print(f"  Block {src_block}→{src_block + 1}: compressing...", flush=True)
        student = _compress_single_block(teacher, corpus, src_block, SEED)
        student.eval()

        row = run_transition_per_head(float_model, student, corpus, src_block)
        all_transitions.append(row)
        heatmap_data[src_block, :] = row["amp_fracs"]

        print(
            f"    top3_frac={row['top3_frac']:.3f}  max_frac={row['max_frac']:.3f}  "
            f"gini={row['gini']:.3f}  delta_L={row['delta_L_mean']:.4f}"
        )

    # ---------------------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------------------
    layers_concentrated = sum(1 for r in all_transitions if r["top3_frac"] >= 0.70)
    layers_uniform = sum(1 for r in all_transitions if r["max_frac"] <= 0.15)

    if layers_concentrated >= 8:
        verdict = "CONFIRMED"
        verdict_reason = f"top-3 >= 70% in {layers_concentrated}/11 layers"
    elif layers_uniform >= 8:
        verdict = "REFUTED"
        verdict_reason = f"max per head <= 15% in {layers_uniform}/11 layers"
    else:
        verdict = "INCONCLUSIVE"
        verdict_reason = f"concentrated={layers_concentrated}/11, uniform={layers_uniform}/11"

    print(f"\n=== VERDICT: {verdict} ({verdict_reason}) ===")

    # ---------------------------------------------------------------------------
    # Optional ablation on block 6 (middle) — representative
    # ---------------------------------------------------------------------------
    print("\nRunning head ablation on block 6 (PPL deltas)...")
    ppl_deltas_6 = _ablation_head_importance(float_model, teacher, corpus, block_id=6)
    print(f"  Block 6 PPL deltas: {[f'{d:.2f}' for d in ppl_deltas_6]}")

    # Spearman correlation between amp_fracs and ppl_deltas for block 6 (transition 5→6)
    # transition src=5 measures block 6 attn
    amp_fracs_6 = None
    for r in all_transitions:
        if r["target_block"] == 6:
            amp_fracs_6 = r["amp_fracs"]
            break

    spearman_rho = None
    if amp_fracs_6 is not None:
        from scipy.stats import spearmanr
        rho, p = spearmanr(amp_fracs_6, ppl_deltas_6)
        spearman_rho = {"rho": float(rho), "p": float(p)}
        print(f"  Spearman rho (amp vs ppl_delta, block 6): rho={rho:.3f}  p={p:.4f}")

    # ---------------------------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------------------------
    output = {
        "config": {
            "seed": SEED,
            "eval_seqs": EVAL_SEQS,
            "seq_len": SEQ_LEN,
            "hidden_fraction": HIDDEN_FRACTION,
            "polish_steps": POLISH_STEPS,
            "n_heads": N_HEADS,
            "head_dim": HEAD_DIM,
        },
        "baseline_ppl": base_ppl,
        "transitions": all_transitions,
        "summary": {
            "layers_concentrated": layers_concentrated,
            "layers_uniform": layers_uniform,
            "verdict": verdict,
            "verdict_reason": verdict_reason,
        },
        "ablation_block6": {
            "ppl_deltas": ppl_deltas_6,
            "spearman": spearman_rho,
        },
    }

    json_path = RESULTS_DIR / "cycle_113_head_amplification.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    # 1. Heatmap: rows=transition (upstream block L), cols=head
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="hot_r", vmin=0, vmax=0.3)
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Upstream Block (FFN compressed)")
    ax.set_title("Per-Head Amplification Share: FFN Compression → Downstream Attention\nGPT-2, Cycle 113")
    ax.set_xticks(range(N_HEADS))
    ax.set_yticks(range(N_TRANSITIONS))
    ax.set_yticklabels([f"Block {i}→{i+1}" for i in range(N_TRANSITIONS)])
    plt.colorbar(im, ax=ax, label="Amplification share (fraction of total)")
    heatmap_path = RESULTS_DIR / "cycle_113_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved heatmap: {heatmap_path}")

    # 2. Summary: top3_frac and max_frac per layer
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    xs = range(N_TRANSITIONS)
    top3s = [r["top3_frac"] for r in all_transitions]
    maxs = [r["max_frac"] for r in all_transitions]

    axes[0].bar(xs, top3s, color="steelblue")
    axes[0].axhline(0.70, color="red", linestyle="--", label="70% threshold")
    axes[0].axhline(1 / 4, color="gray", linestyle=":", label="Null (3×uniform=25%)")
    axes[0].set_xlabel("Transition (source block)")
    axes[0].set_ylabel("Top-3 Heads Share")
    axes[0].set_title("Top-3 Head Amplification Share")
    axes[0].set_xticks(list(xs))
    axes[0].set_xticklabels([str(i) for i in range(N_TRANSITIONS)])
    axes[0].legend()

    axes[1].bar(xs, maxs, color="salmon")
    axes[1].axhline(0.15, color="red", linestyle="--", label="15% threshold (refuted)")
    axes[1].axhline(1 / 12, color="gray", linestyle=":", label="Null (uniform=8.3%)")
    axes[1].set_xlabel("Transition (source block)")
    axes[1].set_ylabel("Max Head Share")
    axes[1].set_title("Max Head Amplification Share")
    axes[1].set_xticks(list(xs))
    axes[1].set_xticklabels([str(i) for i in range(N_TRANSITIONS)])
    axes[1].legend()

    summary_path = RESULTS_DIR / "cycle_113_summary.png"
    plt.suptitle(f"Cycle 113 — {verdict}: {verdict_reason}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150)
    plt.close()
    print(f"Saved summary plot: {summary_path}")

    # 3. Top-10 amplifying heads globally
    # Build (transition, head, amp_frac) tuples
    head_records = []
    for r in all_transitions:
        for h_idx, frac in enumerate(r["amp_fracs"]):
            head_records.append((r["source_block"], h_idx, frac))
    head_records_sorted = sorted(head_records, key=lambda x: x[2], reverse=True)[:10]
    labels = [f"Blk{src}→H{h}" for src, h, _ in head_records_sorted]
    values = [v for _, _, v in head_records_sorted]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(10), values, color="darkorange")
    ax.set_xticks(range(10))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Amplification Share (fraction of layer total)")
    ax.set_title("Top-10 Amplifying Heads Across All Transitions — GPT-2 Cycle 113")
    top10_path = RESULTS_DIR / "cycle_113_top10_heads.png"
    plt.tight_layout()
    plt.savefig(top10_path, dpi=150)
    plt.close()
    print(f"Saved top-10 heads plot: {top10_path}")

    # ---------------------------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------------------------
    manifest_lines = [
        "cycle_113_manifest.txt",
        f"item: 124",
        f"verdict: {verdict}",
        f"verdict_reason: {verdict_reason}",
        f"baseline_ppl: {base_ppl:.2f}",
        f"layers_concentrated (top3>=70%): {layers_concentrated}/11",
        f"layers_uniform (max<=15%): {layers_uniform}/11",
        "",
        "Per-transition summary:",
    ]
    for r in all_transitions:
        manifest_lines.append(
            f"  Block {r['source_block']}→{r['target_block']}: "
            f"top3={r['top3_frac']:.3f}  max={r['max_frac']:.3f}  gini={r['gini']:.3f}"
        )
    manifest_lines += [
        "",
        f"Ablation block 6 spearman_rho: {spearman_rho}",
        "",
        "Files:",
        f"  {json_path}",
        f"  {heatmap_path}",
        f"  {summary_path}",
        f"  {top10_path}",
        f"  experiments/per_head_amplification_census_gpt2.py",
    ]

    manifest_path = RESULTS_DIR / "cycle_113_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest_lines) + "\n")
    print(f"\nSaved manifest: {manifest_path}")
    print(f"\nDone. Verdict: {verdict}")


if __name__ == "__main__":
    main()
