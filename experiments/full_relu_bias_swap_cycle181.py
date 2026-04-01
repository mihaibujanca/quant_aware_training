"""
Cycle 181 / Item 191: Full-model ReLU + per-block bias vector swap — GPT-2

Precompute μ_neg_l = E[W_down @ (GELU(z) ⊙ 1[z≤0])] on 2048 OWT tokens for each
of 12 blocks. Replace all 12 GELU activations with ReLU, add μ_neg_l to c_proj bias.
Evaluate full-model PPL.

Confirmed if: full-model ReLU+bias PPL within +10% of float.
Refuted if: PPL > +50%.
Inconclusive if: +10-50%.
"""

import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
N_SEQS = 200
SEQ_LEN = 1024
BATCH_SIZE = 4
SEED = 42
N_TOKENS_MU = 2048  # tokens for μ_neg estimation
OWT_PATH = "data/openwebtext_sample.txt"
OUT_JSON = "autoresearch/results/cycle_181_relu_bias_swap.json"
OUT_PNG = "autoresearch/results/cycle_181_relu_bias_swap.png"
BASELINE_PPL_LO, BASELINE_PPL_HI = 15.0, 65.0
N_BLOCKS = 12

print(f"Device: {DEVICE}", flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_owt_tokens(tokenizer, n_seqs=N_SEQS, seq_len=SEQ_LEN, seed=SEED):
    with open(OWT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    rng = torch.Generator()
    rng.manual_seed(seed)
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    max_start = len(tokens) - seq_len - 1
    indices = torch.randperm(max_start, generator=rng)[:n_seqs]
    seqs = torch.stack([tokens[i:i + seq_len + 1] for i in sorted(indices.tolist())])
    return seqs  # (n_seqs, seq_len+1)


def eval_ppl(model, seqs, batch_size=BATCH_SIZE):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size].to(DEVICE)
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            out = model(inp)
            logits = out.logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += tgt.numel()
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# Phase 1: Precompute μ_neg_l per block
# ---------------------------------------------------------------------------
def compute_mu_neg(model, seqs, n_tokens=N_TOKENS_MU):
    """
    For each block l, compute:
      E[GELU(z) ⊙ 1[z ≤ 0]] in 3072-dim (intermediate space)
    Then project through W_down to get μ_neg_l in 768-dim.
    """
    N_CHANNELS = 3072  # GPT-2 intermediate size
    accum = {i: torch.zeros(N_CHANNELS, dtype=torch.float64) for i in range(N_BLOCKS)}
    counts = {i: 0 for i in range(N_BLOCKS)}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            z = inp[0]  # pre-activation (batch, seq, 3072)
            activated = out  # GELU(z)
            mask_neg = (z <= 0)
            y_neg = torch.where(mask_neg, activated, torch.zeros_like(activated))
            # Sum over batch and sequence dims
            batch_sum = y_neg.reshape(-1, N_CHANNELS).sum(dim=0)
            n = y_neg.reshape(-1, N_CHANNELS).shape[0]
            accum[idx] += batch_sum.cpu().double()
            counts[idx] += n
        return hook_fn

    for i in range(N_BLOCKS):
        h = model.transformer.h[i].mlp.act.register_forward_hook(make_hook(i))
        hooks.append(h)

    # Run enough sequences to get n_tokens
    n_seqs_needed = max(1, n_tokens // SEQ_LEN + 1)
    model.eval()
    with torch.no_grad():
        for i in range(0, min(n_seqs_needed, len(seqs)), BATCH_SIZE):
            batch = seqs[i:i + BATCH_SIZE, :-1].to(DEVICE)
            model(batch)

    for h in hooks:
        h.remove()

    # Compute μ_neg_l = W_down @ E[y_neg]
    mu_neg = {}
    mu_neg_norms = {}
    for l in range(N_BLOCKS):
        e_y_neg = (accum[l] / counts[l]).float().to(DEVICE)  # (3072,)
        # GPT-2 Conv1D: weight is (in=3072, out=768), forward does x @ W + b
        W_down = model.transformer.h[l].mlp.c_proj.weight.data  # (3072, 768)
        mu = e_y_neg @ W_down  # (768,)
        mu_neg[l] = mu
        mu_neg_norms[l] = mu.norm().item()
        print(f"  Block {l}: ||E[y_neg]||={e_y_neg.norm().item():.4f}, "
              f"||μ_neg_l||={mu_neg_norms[l]:.4f}, tokens={counts[l]}", flush=True)

    return mu_neg, mu_neg_norms


# ---------------------------------------------------------------------------
# Phase 2: Replace GELU with ReLU and add bias
# ---------------------------------------------------------------------------
class ReLUActivation(nn.Module):
    """Drop-in replacement for GELU: just ReLU."""
    def forward(self, z):
        return F.relu(z)


def swap_blocks(model, block_indices, mu_neg):
    """Replace GELU with ReLU and add μ_neg_l to c_proj bias for given blocks."""
    original_acts = {}
    original_biases = {}
    for l in block_indices:
        # Save originals
        original_acts[l] = model.transformer.h[l].mlp.act
        original_biases[l] = model.transformer.h[l].mlp.c_proj.bias.data.clone()
        # Swap activation
        model.transformer.h[l].mlp.act = ReLUActivation()
        # Add μ_neg_l to c_proj bias
        model.transformer.h[l].mlp.c_proj.bias.data += mu_neg[l]
    return original_acts, original_biases


def restore_blocks(model, block_indices, original_acts, original_biases):
    """Restore original GELU activations and biases."""
    for l in block_indices:
        model.transformer.h[l].mlp.act = original_acts[l]
        model.transformer.h[l].mlp.c_proj.bias.data.copy_(original_biases[l])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading model and tokenizer...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    print(f"Activation: {type(model.transformer.h[0].mlp.act).__name__}", flush=True)

    # Load data
    seqs = load_owt_tokens(tokenizer)

    # -------------------------------------------------------------------
    # Baseline PPL
    # -------------------------------------------------------------------
    print("\n=== Baseline PPL ===", flush=True)
    ppl_base = eval_ppl(model, seqs)
    print(f"  Baseline PPL: {ppl_base:.3f}", flush=True)

    if not (BASELINE_PPL_LO <= ppl_base <= BASELINE_PPL_HI):
        print(f"ABORT: Baseline PPL {ppl_base:.3f} outside [{BASELINE_PPL_LO}, {BASELINE_PPL_HI}]")
        return

    # -------------------------------------------------------------------
    # Phase 1: Precompute μ_neg_l
    # -------------------------------------------------------------------
    print("\n=== Phase 1: Precompute μ_neg_l ===", flush=True)
    mu_neg, mu_neg_norms = compute_mu_neg(model, seqs)

    # -------------------------------------------------------------------
    # Phase 2a: Full-model swap (all 12 blocks)
    # -------------------------------------------------------------------
    print("\n=== Phase 2a: Full-model ReLU + bias swap ===", flush=True)
    all_blocks = list(range(N_BLOCKS))
    orig_acts, orig_biases = swap_blocks(model, all_blocks, mu_neg)
    ppl_full = eval_ppl(model, seqs)
    print(f"  Full-model PPL: {ppl_full:.3f}", flush=True)
    restore_blocks(model, all_blocks, orig_acts, orig_biases)

    # -------------------------------------------------------------------
    # Phase 2b: Incremental accumulation (blocks 0..k)
    # -------------------------------------------------------------------
    print("\n=== Phase 2b: Incremental accumulation ===", flush=True)
    ppl_incremental = {}
    for k in range(N_BLOCKS):
        blocks_to_swap = list(range(k + 1))
        orig_a, orig_b = swap_blocks(model, blocks_to_swap, mu_neg)
        ppl_k = eval_ppl(model, seqs)
        ppl_incremental[k] = ppl_k
        delta_pct = (ppl_k - ppl_base) / ppl_base * 100
        print(f"  Blocks 0..{k}: PPL={ppl_k:.3f} (delta={delta_pct:+.1f}%)", flush=True)
        restore_blocks(model, blocks_to_swap, orig_a, orig_b)

    # -------------------------------------------------------------------
    # Phase 2c: Single-block swaps (for comparison with cycle 173)
    # -------------------------------------------------------------------
    print("\n=== Phase 2c: Single-block swaps ===", flush=True)
    ppl_single = {}
    for l in range(N_BLOCKS):
        orig_a, orig_b = swap_blocks(model, [l], mu_neg)
        ppl_l = eval_ppl(model, seqs)
        ppl_single[l] = ppl_l
        delta_pct = (ppl_l - ppl_base) / ppl_base * 100
        print(f"  Block {l} only: PPL={ppl_l:.3f} (delta={delta_pct:+.1f}%)", flush=True)
        restore_blocks(model, [l], orig_a, orig_b)

    # -------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------
    delta_full_pct = (ppl_full - ppl_base) / ppl_base * 100
    if delta_full_pct <= 10:
        verdict = "CONFIRMED"
    elif delta_full_pct > 50:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    print(f"  Baseline PPL: {ppl_base:.3f}", flush=True)
    print(f"  Full-model PPL: {ppl_full:.3f}", flush=True)
    print(f"  Delta: {delta_full_pct:+.1f}%", flush=True)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results = {
        "cycle": 181,
        "item": 191,
        "verdict": verdict,
        "ppl_base": round(ppl_base, 3),
        "ppl_full": round(ppl_full, 3),
        "delta_pct": round(delta_full_pct, 2),
        "mu_neg_norms": {str(k): round(v, 4) for k, v in mu_neg_norms.items()},
        "ppl_incremental": {str(k): round(v, 3) for k, v in ppl_incremental.items()},
        "ppl_single": {str(k): round(v, 3) for k, v in ppl_single.items()},
        "config": {
            "n_seqs": N_SEQS,
            "seq_len": SEQ_LEN,
            "n_tokens_mu": N_TOKENS_MU,
            "seed": SEED,
            "device": DEVICE,
            "n_blocks": N_BLOCKS,
        }
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_JSON}", flush=True)

    # -------------------------------------------------------------------
    # Plot: incremental accumulation + single-block
    # -------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: incremental accumulation
    blocks = list(range(N_BLOCKS))
    ppls = [ppl_incremental[k] for k in blocks]
    deltas = [(p - ppl_base) / ppl_base * 100 for p in ppls]
    ax1.bar(blocks, deltas, color="steelblue", alpha=0.8)
    ax1.axhline(y=10, color="green", linestyle="--", alpha=0.7, label="+10% (confirmed)")
    ax1.axhline(y=50, color="red", linestyle="--", alpha=0.7, label="+50% (refuted)")
    ax1.set_xlabel("Blocks 0..k swapped")
    ax1.set_ylabel("PPL delta (%)")
    ax1.set_title("Incremental accumulation")
    ax1.set_xticks(blocks)
    ax1.legend()

    # Right: single-block swaps
    ppls_single = [ppl_single[l] for l in blocks]
    deltas_single = [(p - ppl_base) / ppl_base * 100 for p in ppls_single]
    ax2.bar(blocks, deltas_single, color="coral", alpha=0.8)
    ax2.axhline(y=2, color="green", linestyle="--", alpha=0.7, label="+2% (cycle 173)")
    ax2.set_xlabel("Block l swapped (alone)")
    ax2.set_ylabel("PPL delta (%)")
    ax2.set_title("Single-block swaps")
    ax2.set_xticks(blocks)
    ax2.legend()

    fig.suptitle(f"Cycle 181: Full-model ReLU+bias swap — {verdict}\n"
                 f"Base={ppl_base:.1f}, Full={ppl_full:.1f} ({delta_full_pct:+.1f}%)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"Plot saved to {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
