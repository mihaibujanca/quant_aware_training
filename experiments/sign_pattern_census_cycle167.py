"""
Cycle 167 / Item 177: Sign-pattern census on actual data — GPT-2 FFN

For each block, record binary sign patterns (z > 0) of pre-GELU activations
on 4096 OWT tokens. Measure:
(a) unique sign pattern count
(b) Hamming distance distribution between patterns
(c) k-means clustering with k={16,32,64,128,256}
(d) K* = min k for <5% within-cluster FFN output variance

Confirmed if: K* <= 64 for >=6/12 blocks.
Refuted if: K* > 256 for majority of blocks.
"""

import json
import math
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
N_TOKENS = 4096
SEQ_LEN = 1024
BATCH_SIZE = 4
SEED = 42
OWT_PATH = "data/openwebtext_sample.txt"
OUT_JSON = "autoresearch/results/cycle_167_sign_census.json"
OUT_PNG = "autoresearch/results/cycle_167_sign_census.png"
N_BLOCKS = 12
K_VALUES = [16, 32, 64, 128, 256]
VARIANCE_THRESHOLD = 0.05  # 5% within-cluster variance ratio

print(f"Device: {DEVICE}", flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_owt_tokens(tokenizer, n_tokens=N_TOKENS, seq_len=SEQ_LEN, seed=SEED):
    """Load enough sequences to get n_tokens total tokens."""
    with open(OWT_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    rng = torch.Generator()
    rng.manual_seed(seed)
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    # Need ceil(n_tokens / seq_len) sequences
    n_seqs = math.ceil(n_tokens / seq_len) + 2  # a few extra
    max_start = len(tokens) - seq_len - 1
    indices = torch.randperm(max_start, generator=rng)[:n_seqs]
    seqs = torch.stack([tokens[i:i+seq_len] for i in sorted(indices.tolist())])
    return seqs  # (n_seqs, seq_len)


# ---------------------------------------------------------------------------
# Hooking: collect pre-GELU z and FFN output y per block
# ---------------------------------------------------------------------------
def collect_activations(model, seqs, device, n_tokens=N_TOKENS):
    """Forward pass, collect pre-GELU z and FFN output y for all blocks."""
    z_per_block = [[] for _ in range(N_BLOCKS)]
    y_per_block = [[] for _ in range(N_BLOCKS)]
    hooks = []

    # Hook mlp.c_fc for pre-GELU z
    for b in range(N_BLOCKS):
        block = model.transformer.h[b]

        def make_z_hook(block_idx):
            def hook_fn(module, input, output):
                # output of c_fc = z (pre-GELU), shape (batch, seq_len, 3072)
                z_per_block[block_idx].append(output.detach().cpu())
            return hook_fn

        def make_y_hook(block_idx):
            def hook_fn(module, input, output):
                # output of mlp = y (FFN output), shape (batch, seq_len, 768)
                y_per_block[block_idx].append(output.detach().cpu())
            return hook_fn

        hooks.append(block.mlp.c_fc.register_forward_hook(make_z_hook(b)))
        hooks.append(block.mlp.register_forward_hook(make_y_hook(b)))

    model.eval()
    with torch.no_grad():
        for i in range(0, len(seqs), BATCH_SIZE):
            batch = seqs[i:i+BATCH_SIZE].to(device)
            model(batch)

    for h in hooks:
        h.remove()

    # Flatten and truncate to n_tokens
    z_all = []
    y_all = []
    for b in range(N_BLOCKS):
        z_cat = torch.cat(z_per_block[b], dim=0)  # (n_seqs, seq_len, 3072)
        z_flat = z_cat.reshape(-1, z_cat.shape[-1])[:n_tokens]  # (n_tokens, 3072)
        z_all.append(z_flat)

        y_cat = torch.cat(y_per_block[b], dim=0)
        y_flat = y_cat.reshape(-1, y_cat.shape[-1])[:n_tokens]  # (n_tokens, 768)
        y_all.append(y_flat)

    return z_all, y_all


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------
def count_unique_patterns(signs):
    """Count unique binary sign patterns. signs: (n_tokens, 3072) bool tensor."""
    # Pack into bytes for efficient comparison
    n = signs.shape[0]
    # Convert to numpy for unique counting
    signs_np = signs.numpy().astype(np.uint8)
    # Pack bits: 3072 / 8 = 384 bytes per pattern
    packed = np.packbits(signs_np, axis=1)  # (n_tokens, 384)
    # Use structured array for unique
    packed_view = packed.view(np.dtype((np.void, packed.shape[1])))
    unique = np.unique(packed_view)
    return len(unique)


def hamming_distances(signs, n_pairs=10000, seed=42):
    """Sample pairs, compute Hamming distances. signs: (n, d) bool tensor."""
    rng = np.random.RandomState(seed)
    n = signs.shape[0]
    idx_a = rng.randint(0, n, size=n_pairs)
    idx_b = rng.randint(0, n, size=n_pairs)
    # Avoid self-pairs
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    # Hamming = number of differing bits
    diffs = (signs[idx_a] != signs[idx_b]).sum(dim=1).numpy()
    return diffs


def kmeans_variance_ratio(signs_np, y_np, k_values, seed=42):
    """Run k-means on sign patterns, compute within-cluster FFN output variance ratio."""
    # Total variance of y
    total_var = np.var(y_np, axis=0).sum()  # sum of per-dim variances

    results = {}
    for k in k_values:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=256,
                             n_init=3, max_iter=100)
        labels = km.fit_predict(signs_np)

        # Within-cluster variance
        within_var = 0.0
        for c in range(k):
            mask = labels == c
            n_c = mask.sum()
            if n_c > 1:
                cluster_var = np.var(y_np[mask], axis=0).sum()
                within_var += cluster_var * n_c

        within_var /= len(labels)  # weighted average
        ratio = within_var / total_var if total_var > 0 else 0.0
        results[k] = {
            "within_var": float(within_var),
            "total_var": float(total_var),
            "ratio": float(ratio),
        }
        print(f"    k={k:4d}: within/total = {ratio:.4f}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load model
    print("Loading GPT-2...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()

    # Load data
    print("Loading OWT tokens...", flush=True)
    seqs = load_owt_tokens(tokenizer)
    print(f"  Loaded {len(seqs)} sequences of length {SEQ_LEN}", flush=True)

    # Collect activations
    print("Collecting activations...", flush=True)
    z_all, y_all = collect_activations(model, seqs, DEVICE)

    results = {"cycle": 167, "item": 177, "n_tokens": N_TOKENS, "per_block": {}}

    all_hamming = []
    all_kstar = []

    for b in range(N_BLOCKS):
        print(f"\nBlock {b}:", flush=True)
        z = z_all[b]  # (n_tokens, 3072)
        y = y_all[b]  # (n_tokens, 768)
        signs = (z > 0)  # bool tensor

        # (a) Unique pattern count
        n_unique = count_unique_patterns(signs)
        frac_positive = signs.float().mean().item()
        print(f"  Unique patterns: {n_unique} / {N_TOKENS}", flush=True)
        print(f"  Fraction positive: {frac_positive:.4f}", flush=True)

        # (b) Hamming distances
        dists = hamming_distances(signs)
        hamming_mean = float(np.mean(dists))
        hamming_std = float(np.std(dists))
        print(f"  Hamming: mean={hamming_mean:.1f}, std={hamming_std:.1f}", flush=True)
        all_hamming.append(dists)

        # (c, d) k-means clustering + variance ratio
        signs_np = signs.numpy().astype(np.float32)
        y_np = y.numpy()
        print("  k-means variance ratios:", flush=True)
        km_results = kmeans_variance_ratio(signs_np, y_np, K_VALUES)

        # K* = min k for ratio < 0.05
        k_star = None
        for k in sorted(K_VALUES):
            if km_results[k]["ratio"] < VARIANCE_THRESHOLD:
                k_star = k
                break
        if k_star is None:
            k_star = ">256"
        print(f"  K* = {k_star}", flush=True)
        all_kstar.append(k_star)

        results["per_block"][str(b)] = {
            "n_unique": n_unique,
            "frac_positive": round(frac_positive, 4),
            "hamming_mean": round(hamming_mean, 1),
            "hamming_std": round(hamming_std, 1),
            "kmeans": {str(k): km_results[k] for k in K_VALUES},
            "k_star": k_star,
        }

    # Summary
    n_confirmed = sum(1 for ks in all_kstar if isinstance(ks, int) and ks <= 64)
    n_refuted = sum(1 for ks in all_kstar if ks == ">256")
    results["summary"] = {
        "k_star_per_block": [str(ks) for ks in all_kstar],
        "n_kstar_le64": n_confirmed,
        "n_kstar_gt256": n_refuted,
    }

    # Verdict
    if n_confirmed >= 6:
        verdict = "CONFIRMED"
    elif n_refuted > 6:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"
    results["verdict"] = verdict
    print(f"\nVerdict: {verdict}", flush=True)
    print(f"  K* <= 64 in {n_confirmed}/12 blocks", flush=True)
    print(f"  K* > 256 in {n_refuted}/12 blocks", flush=True)

    # Save JSON
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_JSON}", flush=True)

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Unique patterns per block
    ax = axes[0, 0]
    unique_counts = [results["per_block"][str(b)]["n_unique"] for b in range(N_BLOCKS)]
    ax.bar(range(N_BLOCKS), unique_counts, color="steelblue")
    ax.set_xlabel("Block")
    ax.set_ylabel("Unique sign patterns")
    ax.set_title(f"Unique sign patterns / {N_TOKENS} tokens")
    ax.set_xticks(range(N_BLOCKS))

    # Plot 2: Hamming distance distributions (overlay)
    ax = axes[0, 1]
    for b in range(N_BLOCKS):
        ax.hist(all_hamming[b], bins=50, alpha=0.3, label=f"B{b}", density=True)
    ax.set_xlabel("Hamming distance")
    ax.set_ylabel("Density")
    ax.set_title("Hamming distance distributions")
    ax.legend(fontsize=6, ncol=3)

    # Plot 3: Variance ratio vs k for each block
    ax = axes[1, 0]
    for b in range(N_BLOCKS):
        ratios = [results["per_block"][str(b)]["kmeans"][str(k)]["ratio"] for k in K_VALUES]
        ax.plot(K_VALUES, ratios, marker="o", markersize=3, label=f"B{b}")
    ax.axhline(y=VARIANCE_THRESHOLD, color="red", linestyle="--", label="5% threshold")
    ax.set_xlabel("k (clusters)")
    ax.set_ylabel("Within-cluster / total variance")
    ax.set_title("Variance ratio vs k")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=6, ncol=3)

    # Plot 4: K* per block
    ax = axes[1, 1]
    kstar_numeric = []
    colors = []
    for ks in all_kstar:
        if isinstance(ks, int):
            kstar_numeric.append(ks)
            colors.append("green" if ks <= 64 else "orange")
        else:
            kstar_numeric.append(512)  # visual placeholder for >256
            colors.append("red")
    ax.bar(range(N_BLOCKS), kstar_numeric, color=colors)
    ax.axhline(y=64, color="green", linestyle="--", alpha=0.5, label="K*=64 threshold")
    ax.axhline(y=256, color="red", linestyle="--", alpha=0.5, label="K*=256 threshold")
    ax.set_xlabel("Block")
    ax.set_ylabel("K*")
    ax.set_title("K* per block (red = >256)")
    ax.set_xticks(range(N_BLOCKS))
    ax.legend(fontsize=8)

    plt.suptitle(f"Cycle 167: Sign-pattern census — {verdict}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
