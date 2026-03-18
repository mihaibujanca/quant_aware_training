"""Cycle 022: Attention head geometry census — GPT-2.

For each of 144 heads (12 layers × 12 heads):
  a. effective rank = nuclear_norm(A_mean) / spectral_norm(A_mean)
  b. head output norm variation = std of ||h_out||_2 across tokens
  c. pairwise CKA within same layer

Prunability score = (1 - eff_rank_norm) * (1 - variation_norm) * redundancy

Intervention: zero top-K heads via zeroing c_proj rows. Measure perplexity at K=5,10,20,30.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aleph.lm_geometric_compression import (
    ProbeCorpusSpec,
    TeacherLoadSpec,
    build_probe_corpus,
    load_nanochat_teacher,
    _iter_lm_batches,
)

RESULTS_DIR = Path("autoresearch/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_LAYERS = 12
N_HEADS = 12
D_HEAD = 64  # 768 / 12
N_PROBE_SEQ = 32
SEQ_LEN = 128
PRUNE_KS = [5, 10, 20, 30]
HOLDOUT_BATCH = 8


# ── Load model ──────────────────────────────────────────────────────────────

def load_model():
    spec = TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2")
    bundle = load_nanochat_teacher(spec)
    # Force eager attention so attn_weights are returned
    bundle.model.config._attn_implementation = "eager"
    for layer in bundle.model.transformer.h:
        layer.attn.config._attn_implementation = "eager"
    return bundle


# ── Probe corpus ─────────────────────────────────────────────────────────────

def get_corpus(bundle):
    spec = ProbeCorpusSpec(
        sequence_length=SEQ_LEN,
        probe_train_sequences=N_PROBE_SEQ,
        probe_holdout_sequences=N_PROBE_SEQ,
        eval_val_sequences=16,
        text_path="data/shakespeare.txt",
        dataset_name="shakespeare",
        tokenizer=bundle.tokenizer,
    )
    return build_probe_corpus(spec)


# ── Collect attention weights and head outputs ────────────────────────────────

def collect_head_data(model, probe_seqs: torch.Tensor, device: torch.device):
    """
    Returns:
      attn_mean[l][h]: mean attention matrix [T, T] (averaged over sequences)
      head_out_all[l][h]: all head output tokens [N_tok, D_HEAD]
    """
    # attn_sum[l][h]: accumulated [T, T], attn_count: number of seqs
    # T is determined from first batch; sequences have SEQ_LEN tokens as input_ids
    attn_sum = [[None for _ in range(N_HEADS)] for _ in range(N_LAYERS)]
    attn_count = [0] * N_LAYERS
    head_out_lists = [[[] for _ in range(N_HEADS)] for _ in range(N_LAYERS)]

    # Hooks: capture attention output from each layer's attn module
    # The attn forward with eager mode returns (attn_output [B,T,768], attn_weights [B,H,T,T])
    # attn_output before c_proj has shape [B, H, T, D_HEAD] — capture via hook on c_proj input.
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, args):
            # args[0]: [B, T, 768] = concatenated head outputs going into c_proj
            captured[f"cprojin_{layer_idx}"] = args[0].detach().cpu()
        return hook_fn

    def make_attn_hook(layer_idx):
        def attn_hook(module, inp, output):
            # output: (attn_output [B,T,768], attn_weights [B,H,T,T] or None)
            weights = output[1]  # [B, H, T, T]
            if weights is not None:
                captured[f"weights_{layer_idx}"] = weights.detach().cpu()
        return attn_hook

    hooks = []
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.attn.register_forward_hook(make_attn_hook(i)))
        hooks.append(block.attn.c_proj.register_forward_pre_hook(make_hook(i)))

    try:
        with torch.no_grad():
            for seq_idx in range(min(N_PROBE_SEQ, probe_seqs.shape[0])):
                ids = probe_seqs[seq_idx:seq_idx+1].to(device)  # [1, T]
                _ = model(ids)

                for i in range(N_LAYERS):
                    w_key = f"weights_{i}"
                    cp_key = f"cprojin_{i}"
                    if w_key in captured and cp_key in captured:
                        weights = captured[w_key]  # [1, H, T, T]
                        cp_in = captured[cp_key]   # [1, T, 768]
                        T = weights.shape[-1]
                        # Accumulate mean attention matrix per head
                        for h in range(N_HEADS):
                            w_h = weights[0, h].float()  # [T, T]
                            if attn_sum[i][h] is None:
                                attn_sum[i][h] = w_h.clone()
                            else:
                                # If T differs between seqs (shouldn't for fixed-length probe), use min
                                Tp = min(attn_sum[i][h].shape[0], T)
                                attn_sum[i][h][:Tp, :Tp] += w_h[:Tp, :Tp]
                        attn_count[i] += 1
                        # Head outputs: cp_in reshaped to [T, H, D_HEAD]
                        ho = cp_in[0].view(-1, N_HEADS, D_HEAD)  # [T, H, D_HEAD]
                        for h in range(N_HEADS):
                            head_out_lists[i][h].append(ho[:, h, :])  # [T, D_HEAD]

                # Clear captured
                captured.clear()
    finally:
        for h in hooks:
            h.remove()

    # Compute means
    attn_mean = []
    for i in range(N_LAYERS):
        layer_means = []
        cnt = max(attn_count[i], 1)
        for h in range(N_HEADS):
            if attn_sum[i][h] is not None:
                layer_means.append(attn_sum[i][h] / cnt)
            else:
                layer_means.append(torch.eye(SEQ_LEN))
        attn_mean.append(layer_means)

    # Concatenate head outputs
    head_out_all = []
    for i in range(N_LAYERS):
        layer_outs = []
        for h in range(N_HEADS):
            if head_out_lists[i][h]:
                layer_outs.append(torch.cat(head_out_lists[i][h], dim=0).float())
            else:
                layer_outs.append(torch.zeros(1, D_HEAD))
        head_out_all.append(layer_outs)

    return attn_mean, head_out_all


# ── Metrics ───────────────────────────────────────────────────────────────────

def effective_rank(A: torch.Tensor) -> float:
    """nuclear_norm / spectral_norm of attention matrix."""
    A = A.float()
    nuc = torch.linalg.matrix_norm(A, "nuc").item()
    spec = torch.linalg.matrix_norm(A, 2).item()
    if spec < 1e-10:
        return 1.0
    return nuc / spec


def head_variation(H: torch.Tensor) -> float:
    """std of per-token L2 norms of head output [N, D_HEAD]."""
    norms = H.norm(dim=-1)  # [N]
    return norms.std().item()


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two matrices [N, d]."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    # HSIC(X,Y) = ||X^T Y||_F^2 / n^2
    n = X.shape[0]
    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y
    hsic_xy = (XtY ** 2).sum() / (n ** 2)
    hsic_xx = (XtX ** 2).sum() / (n ** 2)
    hsic_yy = (YtY ** 2).sum() / (n ** 2)
    denom = math.sqrt(max(float(hsic_xx), 1e-10) * max(float(hsic_yy), 1e-10))
    return float(hsic_xy) / denom


def compute_layer_cka(head_outs: list[torch.Tensor]) -> list[float]:
    """Mean CKA of each head with all other heads in same layer."""
    H = N_HEADS
    cka_matrix = torch.zeros(H, H)
    for h1 in range(H):
        for h2 in range(h1, H):
            if h1 == h2:
                cka_matrix[h1, h2] = 1.0
            else:
                val = linear_cka(head_outs[h1], head_outs[h2])
                cka_matrix[h1, h2] = val
                cka_matrix[h2, h1] = val
    # Mean CKA with other heads
    redundancy = []
    for h in range(H):
        others = [cka_matrix[h, h2].item() for h2 in range(H) if h2 != h]
        redundancy.append(sum(others) / max(len(others), 1))
    return redundancy


# ── Compute perplexity ────────────────────────────────────────────────────────

def compute_perplexity(model, sequences: torch.Tensor, device: torch.device) -> float:
    """Compute perplexity. _iter_lm_batches yields (input_ids[:,:-1], targets[:,1:]).
    We run model on input_ids and compare logits to targets directly."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, targets in _iter_lm_batches(sequences, HOLDOUT_BATCH, device):
            logits = model(input_ids).logits  # [B, T, V]
            # input_ids: [B, T], targets: [B, T], logits: [B, T, V]
            # logits[t] predicts targets[t]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
    return math.exp(min(total_loss / max(total_tokens, 1), 20.0))


# ── Prune heads ───────────────────────────────────────────────────────────────

def zero_heads(model, heads_to_zero: list[tuple[int, int]]):
    """Zero out c_proj rows corresponding to the given (layer, head) pairs."""
    for layer_idx, head_idx in heads_to_zero:
        w = model.transformer.h[layer_idx].attn.c_proj.weight  # [768, 768]
        # In GPT-2 Conv1D, weight shape is [in_features, out_features] = [768, 768]
        # The concatenated head outputs feed into c_proj. Head h occupies columns h*64:(h+1)*64
        # of the input, corresponding to rows h*64:(h+1)*64 in the weight matrix (transposed from Linear).
        # c_proj.weight in Conv1D is [768, 768] where it's applied as x @ w + b
        # So columns 0:64 correspond to head 0, etc.
        start = head_idx * D_HEAD
        end = (head_idx + 1) * D_HEAD
        w.data[start:end, :] = 0.0
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading model...")
    bundle = load_model()
    model = bundle.model
    device = torch.device(bundle.device)

    print("Building probe corpus...")
    corpus = get_corpus(bundle)

    print("Collecting head data over probe sequences...")
    attn_mean, head_out_all = collect_head_data(model, corpus.probe_train, device)

    # --- Compute per-head metrics ---
    print("Computing per-head metrics...")
    eff_ranks = torch.zeros(N_LAYERS, N_HEADS)
    variations = torch.zeros(N_LAYERS, N_HEADS)
    redundancies = torch.zeros(N_LAYERS, N_HEADS)

    for i in range(N_LAYERS):
        layer_redundancy = compute_layer_cka(head_out_all[i])
        for h in range(N_HEADS):
            eff_ranks[i, h] = effective_rank(attn_mean[i][h])
            variations[i, h] = head_variation(head_out_all[i][h])
            redundancies[i, h] = layer_redundancy[h]
        print(f"  Layer {i}: eff_rank range [{eff_ranks[i].min():.2f}, {eff_ranks[i].max():.2f}], "
              f"variation range [{variations[i].min():.3f}, {variations[i].max():.3f}]")

    # --- Prunability score ---
    # Normalize per-layer to [0, 1]
    prunability = torch.zeros(N_LAYERS, N_HEADS)
    for i in range(N_LAYERS):
        er = eff_ranks[i]
        er_norm = (er - er.min()) / (er.max() - er.min() + 1e-8)
        va = variations[i]
        va_norm = (va - va.min()) / (va.max() - va.min() + 1e-8)
        re = redundancies[i]
        prunability[i] = (1 - er_norm) * (1 - va_norm) * re

    # --- Baseline perplexity ---
    print("Computing baseline perplexity...")
    baseline_ppl = compute_perplexity(model, corpus.probe_holdout, device)
    print(f"Baseline PPL: {baseline_ppl:.3f}")

    # --- Intervention: zero top-K heads ---
    # Flatten prunability to get global ranking
    flat_scores = [(prunability[i, h].item(), i, h) for i in range(N_LAYERS) for h in range(N_HEADS)]
    flat_scores.sort(reverse=True)
    print(f"Top 10 most prunable heads: {[(i,h) for _,i,h in flat_scores[:10]]}")

    intervention_results = {}
    for K in PRUNE_KS:
        print(f"Pruning top-{K} heads...")
        pruned_model = copy.deepcopy(model)
        heads_to_zero = [(i, h) for _, i, h in flat_scores[:K]]
        zero_heads(pruned_model, heads_to_zero)
        ppl_k = compute_perplexity(pruned_model, corpus.probe_holdout, device)
        rel_change = (ppl_k - baseline_ppl) / baseline_ppl
        intervention_results[K] = {"ppl": ppl_k, "rel_change": rel_change}
        print(f"  K={K}: PPL={ppl_k:.3f}, rel_change={rel_change:+.3f} ({rel_change*100:+.1f}%)")
        del pruned_model

    # --- Evaluation against criteria ---
    low_rank_threshold = 0.3 * D_HEAD  # 19.2
    n_low_rank = int((eff_ranks < low_rank_threshold).sum().item())
    frac_low_rank = n_low_rank / (N_LAYERS * N_HEADS)
    ppl_change_k10 = intervention_results[10]["rel_change"]

    print(f"\n=== EVALUATION ===")
    print(f"Heads with eff_rank < {low_rank_threshold:.1f}: {n_low_rank}/{N_LAYERS*N_HEADS} = {frac_low_rank:.1%}")
    print(f"PPL change at K=10: {ppl_change_k10:+.3f} ({ppl_change_k10*100:+.1f}%)")
    ppl_change_k5 = intervention_results[5]["rel_change"]

    if frac_low_rank >= 0.20 and ppl_change_k10 <= 0.05:
        verdict = "CONFIRMED"
    elif frac_low_rank < 0.10 and ppl_change_k5 > 0.15:  # ALL high rank + zeroing 5 causes >15% increase
        verdict = "REFUTED"
    elif frac_low_rank >= 0.20 and ppl_change_k10 > 0.05:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "INCONCLUSIVE"

    print(f"Verdict: {verdict}")

    # --- Save results JSON ---
    results = {
        "baseline_ppl": baseline_ppl,
        "eff_ranks": eff_ranks.tolist(),
        "variations": variations.tolist(),
        "redundancies": redundancies.tolist(),
        "prunability": prunability.tolist(),
        "n_low_rank_heads": n_low_rank,
        "frac_low_rank_heads": frac_low_rank,
        "low_rank_threshold": low_rank_threshold,
        "interventions": {str(k): v for k, v in intervention_results.items()},
        "top10_prunable": [(int(i), int(h)) for _, i, h in flat_scores[:10]],
        "verdict": verdict,
    }
    out_json = RESULTS_DIR / "cycle_022_head_census.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved: {out_json}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(eff_ranks.numpy(), aspect="auto", origin="upper")
    axes[0].set_title("Effective Rank (nuc/spec)")
    axes[0].set_xlabel("Head"); axes[0].set_ylabel("Layer")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(variations.numpy(), aspect="auto", origin="upper")
    axes[1].set_title("Head Output Norm Variation (std)")
    axes[1].set_xlabel("Head"); axes[1].set_ylabel("Layer")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(prunability.numpy(), aspect="auto", origin="upper")
    axes[2].set_title("Prunability Score")
    axes[2].set_xlabel("Head"); axes[2].set_ylabel("Layer")
    plt.colorbar(im2, ax=axes[2])

    # Mark top-10 prunable heads
    for rank_k, (_, li, hi) in enumerate(flat_scores[:10]):
        axes[2].text(hi, li, str(rank_k+1), ha="center", va="center",
                     color="red", fontsize=7, fontweight="bold")

    plt.suptitle(f"GPT-2 Attention Head Census | Baseline PPL={baseline_ppl:.2f} | Verdict: {verdict}")
    plt.tight_layout()
    out_png = RESULTS_DIR / "cycle_022_head_census.png"
    plt.savefig(str(out_png), dpi=120, bbox_inches="tight")
    print(f"Saved: {out_png}")

    return results


if __name__ == "__main__":
    main()
