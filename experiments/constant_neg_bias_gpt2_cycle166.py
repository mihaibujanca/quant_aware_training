"""
Cycle 166 / Item 176: Functional role of constant negative bias — GPT-2 FFN

Test if the GELU negative regime is functionally just a bias:
(a) Compute b_leak = E[y_neg] per block on 4096 tokens
(b) Replace GELU → ReLU + b_leak (kill token-dependent negative, keep mean)
(c) Measure per-block and full-model PPL vs float baseline

Confirmed if ReLU + b_leak within +5% PPL in >=8/12 blocks.
Refuted if >+30% PPL. Inconclusive if +5-30%.
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
OWT_PATH = "data/openwebtext_sample.txt"
OUT_JSON = "autoresearch/results/cycle_166_neg_bias.json"
OUT_PNG = "autoresearch/results/cycle_166_neg_bias.png"
BASELINE_PPL_LO, BASELINE_PPL_HI = 15.0, 65.0
N_BLOCKS = 12
N_TOKENS_BLEAK = 4096  # tokens for b_leak estimation

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
    seqs = torch.stack([tokens[i:i+seq_len+1] for i in sorted(indices.tolist())])
    return seqs  # (n_seqs, seq_len+1)


def eval_ppl(model, seqs, batch_size=BATCH_SIZE):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size].to(DEVICE)
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
# b_leak computation
# ---------------------------------------------------------------------------
def compute_b_leak(model, seqs, n_tokens=N_TOKENS_BLEAK):
    """Compute b_leak = E[y_neg] per block. y_neg = gelu(z) where z <= 0."""
    b_leak = {}
    accum = {i: None for i in range(N_BLOCKS)}
    counts = {i: 0 for i in range(N_BLOCKS)}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            z = inp[0]  # pre-activation
            activated = out  # gelu(z)
            mask_neg = (z <= 0)
            y_neg = torch.where(mask_neg, activated, torch.zeros_like(activated))
            # Sum over batch and sequence dims → (3072,)
            batch_sum = y_neg.sum(dim=0)  # (seq, 3072) or just (3072,)
            if batch_sum.dim() > 1:
                batch_sum = batch_sum.sum(dim=0)  # (3072,)
            n = z.shape[0] * (z.shape[1] if z.dim() == 3 else 1)
            if accum[idx] is None:
                accum[idx] = batch_sum.cpu().float()
            else:
                accum[idx] += batch_sum.cpu().float()
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
            batch = seqs[i:i+BATCH_SIZE, :-1].to(DEVICE)
            model(batch)

    for h in hooks:
        h.remove()

    for i in range(N_BLOCKS):
        b_leak[i] = (accum[i] / counts[i]).to(DEVICE)
        print(f"  Block {i}: b_leak mean={b_leak[i].mean().item():.6f}, "
              f"norm={b_leak[i].norm().item():.4f}, tokens={counts[i]}", flush=True)

    return b_leak


# ---------------------------------------------------------------------------
# ReLU + b_leak wrapper
# ---------------------------------------------------------------------------
class ReLUPlusBiasWrapper(nn.Module):
    """Replace GELU(z) with ReLU(z) + b_leak (constant bias from negative regime mean)."""
    def __init__(self, b_leak_vec):
        super().__init__()
        self.register_buffer("b_leak", b_leak_vec)

    def forward(self, z):
        return F.relu(z) + self.b_leak


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading model and tokenizer...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(DEVICE)
    model.eval()
    print(f"Activation class: {type(model.transformer.h[0].mlp.act).__name__}", flush=True)

    # Save original activations
    original_acts = {i: model.transformer.h[i].mlp.act for i in range(N_BLOCKS)}

    # -------------------------------------------------------------------
    # Step 1: Baseline PPL
    # -------------------------------------------------------------------
    print("\n=== Step 1: Baseline PPL ===", flush=True)
    seqs = load_owt_tokens(tokenizer)
    ppl_base = eval_ppl(model, seqs)
    print(f"  Baseline PPL: {ppl_base:.3f}", flush=True)

    if not (BASELINE_PPL_LO <= ppl_base <= BASELINE_PPL_HI):
        print(f"ABORT: Baseline PPL {ppl_base:.3f} outside [{BASELINE_PPL_LO}, {BASELINE_PPL_HI}]")
        return

    # -------------------------------------------------------------------
    # Step 2: Compute b_leak per block
    # -------------------------------------------------------------------
    print("\n=== Step 2: Computing b_leak per block ===", flush=True)
    b_leak = compute_b_leak(model, seqs)

    # Restore originals after b_leak computation (hooks already removed)
    for i in range(N_BLOCKS):
        model.transformer.h[i].mlp.act = original_acts[i]

    # -------------------------------------------------------------------
    # Step 3: Per-block ReLU + b_leak eval
    # -------------------------------------------------------------------
    print("\n=== Step 3: Per-block ReLU + b_leak PPL ===", flush=True)
    per_block_ppl = {}
    per_block_delta_pct = {}

    for block_idx in range(N_BLOCKS):
        # Install wrapper for this block only
        model.transformer.h[block_idx].mlp.act = ReLUPlusBiasWrapper(b_leak[block_idx])

        ppl = eval_ppl(model, seqs)
        delta_pct = (ppl - ppl_base) / ppl_base * 100.0
        per_block_ppl[block_idx] = ppl
        per_block_delta_pct[block_idx] = delta_pct
        print(f"  Block {block_idx:2d}: PPL={ppl:.3f}  delta={delta_pct:+.2f}%", flush=True)

        # Restore
        model.transformer.h[block_idx].mlp.act = original_acts[block_idx]

    # -------------------------------------------------------------------
    # Step 4: Full-model ReLU + b_leak eval
    # -------------------------------------------------------------------
    print("\n=== Step 4: Full-model ReLU + b_leak PPL ===", flush=True)
    for i in range(N_BLOCKS):
        model.transformer.h[i].mlp.act = ReLUPlusBiasWrapper(b_leak[i])

    ppl_full = eval_ppl(model, seqs)
    delta_full_pct = (ppl_full - ppl_base) / ppl_base * 100.0
    print(f"  Full-model PPL: {ppl_full:.3f}  delta={delta_full_pct:+.2f}%", flush=True)

    # Restore
    for i in range(N_BLOCKS):
        model.transformer.h[i].mlp.act = original_acts[i]

    # -------------------------------------------------------------------
    # Step 5: Verdict
    # -------------------------------------------------------------------
    n_within_5 = sum(1 for d in per_block_delta_pct.values() if d <= 5.0)
    n_above_30 = sum(1 for d in per_block_delta_pct.values() if d > 30.0)
    max_delta = max(per_block_delta_pct.values())
    min_delta = min(per_block_delta_pct.values())
    mean_delta = sum(per_block_delta_pct.values()) / N_BLOCKS

    if n_within_5 >= 8:
        verdict = "CONFIRMED"
    elif n_above_30 > 0 or delta_full_pct > 30.0:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    print(f"  Blocks within +5%: {n_within_5}/12", flush=True)
    print(f"  Blocks above +30%: {n_above_30}/12", flush=True)
    print(f"  Mean delta: {mean_delta:+.2f}%", flush=True)
    print(f"  Full-model delta: {delta_full_pct:+.2f}%", flush=True)

    # -------------------------------------------------------------------
    # Step 6: Save results
    # -------------------------------------------------------------------
    results = {
        "cycle": 166,
        "item": 176,
        "verdict": verdict,
        "ppl_base": round(ppl_base, 3),
        "ppl_full_relu_bleak": round(ppl_full, 3),
        "delta_full_pct": round(delta_full_pct, 2),
        "n_within_5pct": n_within_5,
        "n_above_30pct": n_above_30,
        "mean_delta_pct": round(mean_delta, 2),
        "max_delta_pct": round(max_delta, 2),
        "min_delta_pct": round(min_delta, 2),
        "per_block": {
            str(i): {
                "ppl": round(per_block_ppl[i], 3),
                "delta_pct": round(per_block_delta_pct[i], 2)
            }
            for i in range(N_BLOCKS)
        },
        "b_leak_stats": {
            str(i): {
                "mean": round(b_leak[i].mean().item(), 6),
                "norm": round(b_leak[i].norm().item(), 4)
            }
            for i in range(N_BLOCKS)
        }
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_JSON}", flush=True)

    # -------------------------------------------------------------------
    # Step 7: Plot
    # -------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    blocks = list(range(N_BLOCKS))
    deltas = [per_block_delta_pct[i] for i in blocks]
    colors = ["#2ecc71" if d <= 5.0 else "#e74c3c" if d > 30.0 else "#f39c12" for d in deltas]

    ax1.bar(blocks, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(y=5.0, color="green", linestyle="--", alpha=0.7, label="+5% (confirmed)")
    ax1.axhline(y=30.0, color="red", linestyle="--", alpha=0.7, label="+30% (refuted)")
    ax1.set_xlabel("Block")
    ax1.set_ylabel("PPL delta (%)")
    ax1.set_title(f"Per-block: ReLU + b_leak vs Float\n({n_within_5}/12 within +5%)")
    ax1.set_xticks(blocks)
    ax1.legend(fontsize=8)

    # b_leak norms
    norms = [b_leak[i].norm().item() for i in blocks]
    ax2.bar(blocks, norms, color="#3498db", edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Block")
    ax2.set_ylabel("||b_leak||")
    ax2.set_title("b_leak L2 norm per block")
    ax2.set_xticks(blocks)

    fig.suptitle(f"Cycle 166: Constant Negative Bias — {verdict}\n"
                 f"Full-model: PPL {ppl_full:.1f} (delta {delta_full_pct:+.1f}%)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
