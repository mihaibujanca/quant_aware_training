"""
Cycle 164: Positive-channel ablation — GPT-2 FFN
Item 174

Kill the POSITIVE channels: set GELU(z) = 0 where z > 0, keep only the negative leak.
Also: replace positive channels with their token-mean (kill variation, keep magnitude).

Conditions:
  A: where z > 0, set output = 0 (keep only negative leak)
  B: where z > 0, replace with token-mean of gelu(z) (kill token-specific signal)

Evaluations:
  1. Baseline PPL
  2. Full-model Condition A
  3. Full-model Condition B
  4. Per-block Condition A (12 blocks × 1 condition)
  5. Diagnostics: ||y_pos|| vs ||y_neg|| per block (full-model baseline pass)
"""

import json
import math
import os
import torch
import torch.nn as nn
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
OUT_JSON = "autoresearch/results/cycle_164_positive_ablation.json"
OUT_PNG = "autoresearch/results/cycle_164_positive_ablation.png"
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
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += tgt.numel()
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# Hook wrappers
# ---------------------------------------------------------------------------
class PositiveZeroWrapper(nn.Module):
    """Condition A: zero out where z > 0, keep negative leak."""
    def __init__(self, original_act):
        super().__init__()
        self.original_act = original_act

    def forward(self, z):
        activated = self.original_act(z)
        # Keep only where z <= 0 (negative leak region)
        return torch.where(z > 0, torch.zeros_like(activated), activated)


class PositiveMeanWrapper(nn.Module):
    """Condition B: replace positive-region output with token-mean, keep negative leak."""
    def __init__(self, original_act):
        super().__init__()
        self.original_act = original_act

    def forward(self, z):
        activated = self.original_act(z)
        # Compute mean over token dimension (dim=1) per batch and channel
        # activated shape: (B, T, D) or (B, D)
        if activated.dim() == 3:
            pos_mean = activated.mean(dim=1, keepdim=True).expand_as(activated)
        else:
            pos_mean = activated.mean(dim=0, keepdim=True).expand_as(activated)
        return torch.where(z > 0, pos_mean, activated)


def install_hook(model, block_idx, wrapper_class, original_gelus):
    model.transformer.h[block_idx].mlp.act = wrapper_class(original_gelus[block_idx])


def install_hook_all(model, wrapper_class, original_gelus):
    for i in range(N_BLOCKS):
        install_hook(model, i, wrapper_class, original_gelus)


def restore_all(model, original_gelus):
    for i in range(N_BLOCKS):
        model.transformer.h[i].mlp.act = original_gelus[i]


# ---------------------------------------------------------------------------
# Diagnostics: measure ||y_pos|| vs ||y_neg|| per block
# ---------------------------------------------------------------------------
def measure_pos_neg_norms(model, seqs, original_gelus, batch_size=BATCH_SIZE):
    """Measure mean ||y_pos|| and ||y_neg|| per block on a forward pass."""
    pos_norms = [[] for _ in range(N_BLOCKS)]
    neg_norms = [[] for _ in range(N_BLOCKS)]
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            # inp[0] is z (pre-activation input), out is already gelu(z)
            z = inp[0]
            activated = out  # use the already-computed output, avoid recursion
            mask_pos = (z > 0)
            y_pos = torch.where(mask_pos, activated, torch.zeros_like(activated))
            y_neg = torch.where(~mask_pos, activated, torch.zeros_like(activated))
            pos_norms[idx].append(y_pos.norm(dim=-1).mean().item())
            neg_norms[idx].append(y_neg.norm(dim=-1).mean().item())
        return hook_fn

    for i in range(N_BLOCKS):
        h = model.transformer.h[i].mlp.act.register_forward_hook(
            make_hook(i)
        )
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for i in range(0, min(len(seqs), N_SEQS), batch_size):
            batch = seqs[i:i+batch_size, :-1].to(DEVICE)
            model(batch)

    for h in hooks:
        h.remove()

    mean_pos = [sum(v) / len(v) for v in pos_norms]
    mean_neg = [sum(v) / len(v) for v in neg_norms]
    return mean_pos, mean_neg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading model and tokenizer...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(DEVICE)
    model.eval()

    act_class = type(model.transformer.h[0].mlp.act).__name__
    print(f"Activation class: {act_class}", flush=True)

    # Save original activations
    original_gelus = {i: model.transformer.h[i].mlp.act for i in range(N_BLOCKS)}

    # -------------------------------------------------------------------
    # Step 1: Baseline
    # -------------------------------------------------------------------
    print("\n=== Step 1: Baseline PPL ===", flush=True)
    seqs = load_owt_tokens(tokenizer)
    ppl_base = eval_ppl(model, seqs)
    print(f"  Baseline PPL: {ppl_base:.3f}", flush=True)

    if not (BASELINE_PPL_LO <= ppl_base <= BASELINE_PPL_HI):
        print(f"ABORT: Baseline PPL {ppl_base:.3f} outside [{BASELINE_PPL_LO}, {BASELINE_PPL_HI}]")
        return

    # -------------------------------------------------------------------
    # Step 2: Diagnostics — pos/neg norms per block
    # -------------------------------------------------------------------
    print("\n=== Step 2: Diagnostics (pos/neg norms) ===", flush=True)
    mean_pos_norms, mean_neg_norms = measure_pos_neg_norms(model, seqs, original_gelus)
    for i in range(N_BLOCKS):
        ratio = mean_pos_norms[i] / (mean_neg_norms[i] + 1e-8)
        print(f"  Block {i:2d}: ||y_pos||={mean_pos_norms[i]:.4f}  ||y_neg||={mean_neg_norms[i]:.4f}  ratio={ratio:.3f}", flush=True)

    # -------------------------------------------------------------------
    # Step 3: Full-model Condition A (zero positive)
    # -------------------------------------------------------------------
    print("\n=== Step 3: Full-model Condition A (zero positive) ===", flush=True)
    install_hook_all(model, PositiveZeroWrapper, original_gelus)
    ppl_A_full = eval_ppl(model, seqs)
    restore_all(model, original_gelus)
    delta_A = (ppl_A_full - ppl_base) / ppl_base * 100
    print(f"  Condition A full-model PPL: {ppl_A_full:.3f}  delta={delta_A:+.1f}%", flush=True)

    # -------------------------------------------------------------------
    # Step 4: Full-model Condition B (mean replacement)
    # -------------------------------------------------------------------
    print("\n=== Step 4: Full-model Condition B (token-mean replacement) ===", flush=True)
    install_hook_all(model, PositiveMeanWrapper, original_gelus)
    ppl_B_full = eval_ppl(model, seqs)
    restore_all(model, original_gelus)
    delta_B = (ppl_B_full - ppl_base) / ppl_base * 100
    print(f"  Condition B full-model PPL: {ppl_B_full:.3f}  delta={delta_B:+.1f}%", flush=True)

    # -------------------------------------------------------------------
    # Step 5: Per-block Condition A
    # -------------------------------------------------------------------
    print("\n=== Step 5: Per-block Condition A ===", flush=True)
    per_block_ppl_A = []
    per_block_delta_A = []
    for i in range(N_BLOCKS):
        install_hook(model, i, PositiveZeroWrapper, original_gelus)
        ppl_i = eval_ppl(model, seqs)
        restore_all(model, original_gelus)
        delta_i = (ppl_i - ppl_base) / ppl_base * 100
        per_block_ppl_A.append(ppl_i)
        per_block_delta_A.append(delta_i)
        print(f"  Block {i:2d}: PPL={ppl_i:.3f}  delta={delta_i:+.1f}%", flush=True)

    # -------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------
    print("\n=== Verdict ===", flush=True)
    if delta_A > 100:
        verdict = "CONFIRMED"
    elif delta_A < 20:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    print(f"  Delta A (full-model): {delta_A:+.1f}%  ->  {verdict}", flush=True)

    # -------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------
    results = {
        "cycle": 164,
        "item": 174,
        "device": DEVICE,
        "n_seqs": N_SEQS,
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "ppl_baseline": ppl_base,
        "ppl_A_full": ppl_A_full,
        "ppl_B_full": ppl_B_full,
        "delta_A_pct": delta_A,
        "delta_B_pct": delta_B,
        "per_block_ppl_A": per_block_ppl_A,
        "per_block_delta_A_pct": per_block_delta_A,
        "mean_pos_norms": mean_pos_norms,
        "mean_neg_norms": mean_neg_norms,
        "verdict": verdict,
    }
    os.makedirs("autoresearch/results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved JSON: {OUT_JSON}", flush=True)

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    blocks = list(range(N_BLOCKS))

    # Panel 1: Per-block PPL delta Condition A
    ax = axes[0]
    ax.bar(blocks, per_block_delta_A, color="tomato")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Block index")
    ax.set_ylabel("PPL delta (%)")
    ax.set_title("Per-block: Kill positive channels (Cond A)")
    ax.set_xticks(blocks)

    # Panel 2: Full-model summary
    ax = axes[1]
    conditions = ["Baseline", "Full-A (zero pos)", "Full-B (mean pos)"]
    ppls = [ppl_base, ppl_A_full, ppl_B_full]
    colors = ["steelblue", "tomato", "darkorange"]
    ax.bar(conditions, ppls, color=colors)
    ax.set_ylabel("PPL")
    ax.set_title(f"Full-model conditions\nVerdict: {verdict}")
    for j, (c, p) in enumerate(zip(conditions, ppls)):
        ax.text(j, p + 0.5, f"{p:.1f}", ha="center", va="bottom", fontsize=9)

    # Panel 3: ||y_pos|| vs ||y_neg|| per block
    ax = axes[2]
    x = [b - 0.2 for b in blocks]
    x2 = [b + 0.2 for b in blocks]
    ax.bar(x, mean_pos_norms, width=0.4, label="||y_pos||", color="tomato")
    ax.bar(x2, mean_neg_norms, width=0.4, label="||y_neg||", color="steelblue")
    ax.set_xlabel("Block index")
    ax.set_ylabel("Mean norm")
    ax.set_title("Positive vs Negative channel norms")
    ax.set_xticks(blocks)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    plt.close()
    print(f"  Saved plot: {OUT_PNG}", flush=True)

    print(f"\nDone. Verdict: {verdict}", flush=True)
    return results


if __name__ == "__main__":
    main()
