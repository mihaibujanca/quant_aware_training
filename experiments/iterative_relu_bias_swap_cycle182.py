"""
Cycle 182 / Item 192: Iterative bias recomputation for ReLU swap — GPT-2

Fix for cycle 181's precomputed approach: swap blocks sequentially and recompute
each μ_neg_l on the partially-swapped model. For l=0..11:
  1. Compute μ_neg_l = W_down @ E[GELU(z) ⊙ 1[z≤0]] on current model state
  2. Replace GELU→ReLU at block l
  3. Add μ_neg_l to c_proj bias
  4. Eval progressive PPL

Confirmed if: iterative PPL ≥10 PPL points better than precomputed (42.6).
Refuted if: within 3 PPL of precomputed.
Inconclusive if: 3-10 PPL improvement.
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
OUT_JSON = "autoresearch/results/cycle_182_iterative_relu_swap.json"
OUT_PNG = "autoresearch/results/cycle_182_iterative_relu_swap.png"
BASELINE_PPL_LO, BASELINE_PPL_HI = 15.0, 65.0
N_BLOCKS = 12
N_CHANNELS = 3072  # GPT-2 intermediate size

# Cycle 181 precomputed results for comparison
PRECOMPUTED_PPL = 42.643
PRECOMPUTED_INCREMENTAL = {
    0: 23.476, 1: 23.843, 2: 24.805, 3: 26.056,
    4: 29.529, 5: 32.353, 6: 35.39, 7: 36.742,
    8: 37.906, 9: 38.356, 10: 38.625, 11: 42.643,
}

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
# Compute μ_neg for a single block on current model state
# ---------------------------------------------------------------------------
def compute_mu_neg_single(model, block_idx, seqs, n_tokens=N_TOKENS_MU):
    """
    Compute μ_neg_l = W_down @ E[GELU(z) ⊙ 1[z≤0]] for block block_idx
    on the current model state (upstream blocks may already be ReLU).
    """
    accum = torch.zeros(N_CHANNELS, dtype=torch.float64)
    count = 0

    def hook_fn(module, inp, out):
        nonlocal accum, count
        z = inp[0]  # pre-activation (batch, seq, 3072)
        activated = out  # GELU(z) or ReLU(z) — whatever the current activation is
        mask_neg = (z <= 0)
        y_neg = torch.where(mask_neg, activated, torch.zeros_like(activated))
        batch_sum = y_neg.reshape(-1, N_CHANNELS).sum(dim=0)
        n = y_neg.reshape(-1, N_CHANNELS).shape[0]
        accum += batch_sum.cpu().double()
        count += n

    h = model.transformer.h[block_idx].mlp.act.register_forward_hook(hook_fn)

    n_seqs_needed = max(1, n_tokens // SEQ_LEN + 1)
    model.eval()
    with torch.no_grad():
        for i in range(0, min(n_seqs_needed, len(seqs)), BATCH_SIZE):
            batch = seqs[i:i + BATCH_SIZE, :-1].to(DEVICE)
            model(batch)

    h.remove()

    e_y_neg = (accum / count).float().to(DEVICE)  # (3072,)
    # GPT-2 Conv1D: weight is (in=3072, out=768), forward does x @ W + b
    W_down = model.transformer.h[block_idx].mlp.c_proj.weight.data  # (3072, 768)
    mu = e_y_neg @ W_down  # (768,)
    return mu, e_y_neg.norm().item(), mu.norm().item()


# ---------------------------------------------------------------------------
# ReLU activation replacement
# ---------------------------------------------------------------------------
class ReLUActivation(nn.Module):
    """Drop-in replacement for GELU: just ReLU."""
    def forward(self, z):
        return F.relu(z)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading model and tokenizer...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    print(f"Activation type: {type(model.transformer.h[0].mlp.act).__name__}", flush=True)

    # Load data
    seqs = load_owt_tokens(tokenizer)

    # -------------------------------------------------------------------
    # Baseline PPL (all GELU, no modifications)
    # -------------------------------------------------------------------
    print("\n=== Baseline PPL ===", flush=True)
    ppl_base = eval_ppl(model, seqs)
    print(f"  Baseline PPL: {ppl_base:.3f}", flush=True)

    if not (BASELINE_PPL_LO <= ppl_base <= BASELINE_PPL_HI):
        print(f"ABORT: Baseline PPL {ppl_base:.3f} outside [{BASELINE_PPL_LO}, {BASELINE_PPL_HI}]")
        return

    # -------------------------------------------------------------------
    # Iterative sequential swap: for l=0..11
    # -------------------------------------------------------------------
    print("\n=== Iterative sequential ReLU+bias swap ===", flush=True)
    ppl_progressive = {}
    mu_neg_norms = {}
    original_biases = {}

    for l in range(N_BLOCKS):
        print(f"\n--- Block {l} ---", flush=True)

        # (a) Compute μ_neg_l on current model state (blocks 0..l-1 already ReLU)
        mu_l, e_y_neg_norm, mu_norm = compute_mu_neg_single(model, l, seqs)
        mu_neg_norms[l] = mu_norm
        print(f"  ||E[y_neg]||={e_y_neg_norm:.4f}, ||μ_neg_l||={mu_norm:.4f}", flush=True)

        # (b) Save original bias, replace GELU→ReLU
        original_biases[l] = model.transformer.h[l].mlp.c_proj.bias.data.clone()
        model.transformer.h[l].mlp.act = ReLUActivation()

        # (c) Add μ_neg_l to c_proj bias
        model.transformer.h[l].mlp.c_proj.bias.data += mu_l

        # (d) Eval progressive PPL
        ppl_l = eval_ppl(model, seqs)
        ppl_progressive[l] = ppl_l
        delta_base_pct = (ppl_l - ppl_base) / ppl_base * 100
        print(f"  PPL after blocks 0..{l} swapped: {ppl_l:.3f} ({delta_base_pct:+.1f}% vs baseline)", flush=True)

    # -------------------------------------------------------------------
    # Final results
    # -------------------------------------------------------------------
    ppl_iterative = ppl_progressive[N_BLOCKS - 1]
    improvement = PRECOMPUTED_PPL - ppl_iterative
    delta_base_pct = (ppl_iterative - ppl_base) / ppl_base * 100

    if improvement >= 10:
        verdict = "CONFIRMED"
    elif improvement <= 3:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    print(f"  Float baseline PPL: {ppl_base:.3f}", flush=True)
    print(f"  Precomputed (cycle 181) PPL: {PRECOMPUTED_PPL:.3f}", flush=True)
    print(f"  Iterative PPL: {ppl_iterative:.3f}", flush=True)
    print(f"  Improvement over precomputed: {improvement:.3f} PPL points", flush=True)
    print(f"  Delta from baseline: {delta_base_pct:+.1f}%", flush=True)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results = {
        "cycle": 182,
        "item": 192,
        "verdict": verdict,
        "ppl_base": round(ppl_base, 3),
        "ppl_iterative": round(ppl_iterative, 3),
        "ppl_precomputed": PRECOMPUTED_PPL,
        "improvement_ppl_points": round(improvement, 3),
        "delta_base_pct": round(delta_base_pct, 2),
        "mu_neg_norms": {str(k): round(v, 4) for k, v in mu_neg_norms.items()},
        "ppl_progressive": {str(k): round(v, 3) for k, v in ppl_progressive.items()},
        "precomputed_incremental": {str(k): v for k, v in PRECOMPUTED_INCREMENTAL.items()},
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
    # Plot: iterative progressive vs precomputed incremental
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    blocks = list(range(N_BLOCKS))
    ppls_iter = [ppl_progressive[k] for k in blocks]
    ppls_precomp = [PRECOMPUTED_INCREMENTAL[k] for k in blocks]

    ax.plot(blocks, ppls_iter, 'o-', color="steelblue", linewidth=2, markersize=8, label="Iterative (this cycle)")
    ax.plot(blocks, ppls_precomp, 's--', color="coral", linewidth=2, markersize=8, label="Precomputed (cycle 181)")
    ax.axhline(y=ppl_base, color="green", linestyle=":", alpha=0.7, label=f"Float baseline ({ppl_base:.1f})")
    ax.axhline(y=PRECOMPUTED_PPL - 10, color="purple", linestyle="--", alpha=0.5,
               label=f"CONFIRMED threshold ({PRECOMPUTED_PPL - 10:.1f})")
    ax.axhline(y=PRECOMPUTED_PPL - 3, color="orange", linestyle="--", alpha=0.5,
               label=f"REFUTED threshold ({PRECOMPUTED_PPL - 3:.1f})")

    ax.set_xlabel("Blocks 0..k swapped to ReLU+bias", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_xticks(blocks)
    ax.legend(fontsize=10)
    ax.set_title(f"Cycle 182: Iterative vs Precomputed ReLU+bias swap — {verdict}\n"
                 f"Iterative={ppl_iterative:.1f} vs Precomputed={PRECOMPUTED_PPL:.1f} "
                 f"(improvement={improvement:+.1f} PPL)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"Plot saved to {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
