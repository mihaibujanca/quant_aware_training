"""
Cycle 102: Zero-shot GELU→ReLU Swap — GPT-2
Item 107

Evaluates PPL degradation when GELU activations in GPT-2's FFN layers
are replaced with ReLU, without any fine-tuning.

Steps:
1. Baseline PPL (OWT + WikiText-2)
2. Full-model swap (all 12 blocks)
3. Per-block swap (each block independently)
4. Activation sparsity (full-swap model)
5. Correlation with cycle_048 complexity (Spearman rho)
"""

import json
import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 7]
N_SEQS = 200
SEQ_LEN = 1024
OWT_PATH = "data/openwebtext_sample.txt"
COMPLEXITY_PATH = "autoresearch/results/cycle_048_local_complexity_gpt2.json"
OUT_JSON = "autoresearch/results/cycle_102_gelu_relu_swap.json"
OUT_PNG = "autoresearch/results/cycle_102_gelu_relu_swap.png"
BASELINE_PPL_LO, BASELINE_PPL_HI = 15.0, 65.0

print(f"Device: {DEVICE}", flush=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_owt_tokens(tokenizer, n_seqs=N_SEQS, seq_len=SEQ_LEN, seed=42):
    """Load OWT, tokenize, return (n_seqs, seq_len) tensor."""
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


def eval_ppl(model, seqs, batch_size=4):
    """Compute perplexity over seqs (n, seq_len+1). Returns float."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size].to(DEVICE)
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            out = model(inp)
            logits = out.logits  # (B, T, V)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += tgt.numel()
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# Model surgery helpers
# ---------------------------------------------------------------------------
def find_gelu_modules(model):
    """Return list of (block_idx, module_path, parent, attr_name) for all GELU activations."""
    results = []
    for i, block in enumerate(model.transformer.h):
        act = block.mlp.act
        results.append((i, f"transformer.h[{i}].mlp.act", block.mlp, "act"))
    return results


def replace_block_gelu(model, block_idx, activation_class=nn.ReLU):
    """Replace GELU in block_idx with activation_class()."""
    model.transformer.h[block_idx].mlp.act = activation_class()


def restore_block_gelu(model, block_idx, gelu_instances):
    """Restore original GELU for block_idx."""
    model.transformer.h[block_idx].mlp.act = gelu_instances[block_idx]


def save_all_gelus(model):
    """Save and return dict {block_idx: original_act}."""
    return {i: model.transformer.h[i].mlp.act for i in range(len(model.transformer.h))}


# ---------------------------------------------------------------------------
# Sparsity measurement
# ---------------------------------------------------------------------------
def measure_sparsity(model, seqs, batch_size=4):
    """Hook post-activation outputs in full-swap model, compute sparsity per block."""
    n_blocks = len(model.transformer.h)
    zero_counts = [0] * n_blocks
    total_counts = [0] * n_blocks
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            zero_counts[idx] += (out == 0).sum().item()
            total_counts[idx] += out.numel()
        return hook_fn

    for i in range(n_blocks):
        h = model.transformer.h[i].mlp.act.register_forward_hook(make_hook(i))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for i in range(0, min(len(seqs), N_SEQS), batch_size):
            batch = seqs[i:i+batch_size][:, :-1].to(DEVICE)
            model(batch)

    for h in hooks:
        h.remove()

    sparsity = [zero_counts[i] / total_counts[i] if total_counts[i] > 0 else 0.0
                for i in range(n_blocks)]
    return sparsity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading model and tokenizer...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(DEVICE)
    model.eval()

    # Verify GELU class
    sample_act = model.transformer.h[0].mlp.act
    print(f"Activation class: {type(sample_act).__name__}", flush=True)

    # Save original GELUs
    original_gelus = save_all_gelus(model)

    # -------------------------------------------------------------------
    # Step 1: Baseline
    # -------------------------------------------------------------------
    print("\n=== Step 1: Baseline PPL ===", flush=True)
    baseline_ppls_owt = []
    all_seqs = {}
    for seed in SEEDS:
        seqs = load_owt_tokens(tokenizer, n_seqs=N_SEQS, seq_len=SEQ_LEN, seed=seed)
        all_seqs[seed] = seqs
        ppl = eval_ppl(model, seqs)
        baseline_ppls_owt.append(ppl)
        print(f"  Seed {seed}: OWT PPL = {ppl:.2f}", flush=True)

    baseline_mean = np.mean(baseline_ppls_owt)
    baseline_std = np.std(baseline_ppls_owt)
    print(f"  Baseline OWT: {baseline_mean:.2f} ± {baseline_std:.2f}", flush=True)

    if not (BASELINE_PPL_LO <= baseline_mean <= BASELINE_PPL_HI):
        print(f"ABORT: Baseline PPL {baseline_mean:.2f} outside [{BASELINE_PPL_LO}, {BASELINE_PPL_HI}]")
        sys.exit(1)

    # -------------------------------------------------------------------
    # Step 2: Full-model swap
    # -------------------------------------------------------------------
    print("\n=== Step 2: Full-model swap ===", flush=True)
    n_blocks = len(model.transformer.h)
    for i in range(n_blocks):
        replace_block_gelu(model, i, nn.ReLU)

    full_swap_ppls = []
    for seed in SEEDS:
        seqs = all_seqs[seed]
        ppl = eval_ppl(model, seqs)
        full_swap_ppls.append(ppl)
        print(f"  Seed {seed}: full-swap OWT PPL = {ppl:.2f}", flush=True)

    full_swap_mean = np.mean(full_swap_ppls)
    full_swap_std = np.std(full_swap_ppls)
    delta_full = (full_swap_mean - baseline_mean) / baseline_mean * 100.0
    print(f"  Full-swap OWT: {full_swap_mean:.2f} ± {full_swap_std:.2f} (delta={delta_full:+.1f}%)", flush=True)

    # -------------------------------------------------------------------
    # Step 4: Sparsity (while model is fully swapped)
    # -------------------------------------------------------------------
    print("\n=== Step 4: Activation sparsity (full swap) ===", flush=True)
    seed_seqs = all_seqs[SEEDS[0]]  # use first seed
    sparsity = measure_sparsity(model, seed_seqs)
    for i, s in enumerate(sparsity):
        print(f"  Block {i:2d}: sparsity = {s:.4f}", flush=True)

    # -------------------------------------------------------------------
    # Restore model
    # -------------------------------------------------------------------
    for i in range(n_blocks):
        restore_block_gelu(model, i, original_gelus)

    # -------------------------------------------------------------------
    # Step 3: Per-block swap
    # -------------------------------------------------------------------
    print("\n=== Step 3: Per-block swap ===", flush=True)
    per_block_ppls = {}  # block_idx -> list of ppls
    for block_idx in range(n_blocks):
        replace_block_gelu(model, block_idx, nn.ReLU)
        ppls = []
        for seed in SEEDS:
            ppl = eval_ppl(model, all_seqs[seed])
            ppls.append(ppl)
        per_block_ppls[block_idx] = ppls
        mean_ppl = np.mean(ppls)
        delta = (mean_ppl - baseline_mean) / baseline_mean * 100.0
        print(f"  Block {block_idx:2d}: PPL = {mean_ppl:.2f} ± {np.std(ppls):.2f} (delta={delta:+.1f}%)", flush=True)
        # Restore
        restore_block_gelu(model, block_idx, original_gelus)

    per_block_means = [np.mean(per_block_ppls[i]) for i in range(n_blocks)]
    per_block_deltas = [(m - baseline_mean) / baseline_mean * 100.0 for m in per_block_means]

    # -------------------------------------------------------------------
    # Step 5: Correlation with complexity
    # -------------------------------------------------------------------
    print("\n=== Step 5: Correlation with cycle_048 complexity ===", flush=True)
    complexity_data = json.load(open(COMPLEXITY_PATH))
    proxies = complexity_data["proxies_by_layer"]
    p1_frob_std = [proxies[str(i)]["p1_frob_std"] for i in range(n_blocks)]

    rho, pval = spearmanr(per_block_deltas, p1_frob_std)
    print(f"  Spearman rho (per-block delta vs p1_frob_std) = {rho:.4f}, p = {pval:.4f}", flush=True)

    # -------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------
    if delta_full <= 10.0:
        verdict = "CONFIRMED"
    elif delta_full > 30.0:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\nVerdict: {verdict} (delta_full = {delta_full:+.1f}%)", flush=True)

    # -------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------
    results = {
        "cycle": 102,
        "item": 107,
        "seeds": SEEDS,
        "n_seqs": N_SEQS,
        "seq_len": SEQ_LEN,
        "baseline": {
            "owt_ppls": baseline_ppls_owt,
            "owt_mean": float(baseline_mean),
            "owt_std": float(baseline_std),
        },
        "full_swap": {
            "owt_ppls": full_swap_ppls,
            "owt_mean": float(full_swap_mean),
            "owt_std": float(full_swap_std),
            "delta_pct": float(delta_full),
        },
        "per_block": {
            str(i): {
                "ppls": per_block_ppls[i],
                "mean": float(per_block_means[i]),
                "delta_pct": float(per_block_deltas[i]),
            }
            for i in range(n_blocks)
        },
        "sparsity_per_block": {str(i): float(sparsity[i]) for i in range(n_blocks)},
        "correlation": {
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
            "complexity_metric": "p1_frob_std",
        },
        "verdict": verdict,
    }
    os.makedirs("autoresearch/results", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON: {OUT_JSON}", flush=True)

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: per-block PPL delta
    ax = axes[0]
    block_ids = list(range(n_blocks))
    ax.bar(block_ids, per_block_deltas, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(10, color="orange", linewidth=1, linestyle="--", label="+10% threshold")
    ax.axhline(30, color="red", linewidth=1, linestyle="--", label="+30% threshold")
    ax.set_xlabel("Block index")
    ax.set_ylabel("PPL delta (%)")
    ax.set_title(f"Per-block GELU→ReLU PPL impact\nFull-model delta: {delta_full:+.1f}% → {verdict}")
    ax.legend(fontsize=8)
    ax.set_xticks(block_ids)

    # Panel 2: sparsity per block
    ax = axes[1]
    ax.bar(block_ids, sparsity, color="darkorange", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Block index")
    ax.set_ylabel("Fraction zeros (sparsity)")
    ax.set_title(f"Post-ReLU activation sparsity\n(full-swap model)")
    ax.set_xticks(block_ids)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved PNG: {OUT_PNG}", flush=True)

    print("\nDone.", flush=True)
    return results


if __name__ == "__main__":
    main()
