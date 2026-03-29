"""
Cycle 085 — GELU activation pattern diversity — GPT-2 FFN (item 97).
For all 12 blocks: per-token soft activation patterns via GELU'(z),
binary pattern analysis, k-means sweep, centroid reconstruction error.
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from aleph.lm_geometric_compression import (
    TeacherLoadSpec,
    ProbeCorpusSpec,
    load_nanochat_teacher,
    build_probe_corpus,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "autoresearch", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_BLOCKS = 12
N_TOKENS = 2048
N_TRAIN = 1536
N_HOLDOUT = 512
K_VALUES = [8, 16, 32, 64, 128, 256]
RECON_ERROR_THRESH = 0.05  # 5%
BATCH_SIZE = 4

# ── Sanity checks ─────────────────────────────────────────────────────────────
print("=== Sanity checks ===")
for x_val, expected_approx, tol in [(0.0, 0.5, 0.01), (3.0, 1.0, 0.02), (-3.0, -0.012, 0.01)]:
    z = torch.tensor([x_val], requires_grad=True)
    g = F.gelu(z, approximate='tanh')
    gp = torch.autograd.grad(g.sum(), z)[0]
    assert abs(gp.item() - expected_approx) < tol, (
        f"Sanity fail at x={x_val}: got {gp.item():.4f}, expected ~{expected_approx}"
    )
    print(f"  GELU'({x_val}) = {gp.item():.4f}  ok (expected ~{expected_approx})")


def gelu_deriv(z_tensor):
    """Compute GELU'(z) via autograd. z_tensor: (n, d). Returns (n, d) gradient."""
    z = z_tensor.clone().requires_grad_(True)
    g = F.gelu(z, approximate='tanh')
    gp = torch.autograd.grad(g.sum(), z)[0]
    return gp.detach()


# ── Load GPT-2 ────────────────────────────────────────────────────────────────
print("\n=== Loading GPT-2 ===")
teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
model = teacher.model
tokenizer = teacher.tokenizer
model.eval()

# ── Build 2048-token probe corpus ─────────────────────────────────────────────
print("\n=== Building probe corpus ===")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
corpus_parts = []
for fname in ["shakespeare.txt", "wikitext2_valid.txt", "openwebtext_sample.txt", "c4_sample.txt"]:
    spec = ProbeCorpusSpec(
        text_path=os.path.join(DATA, fname),
        probe_train_sequences=4,
        sequence_length=128,
        teacher=teacher,
    )
    part = build_probe_corpus(spec)
    corpus_parts.append(part.probe_train)
    print(f"  {fname}: {part.probe_train.shape}")

all_tokens = torch.cat(corpus_parts, dim=0)  # (16, 128)
flat_tokens = all_tokens.view(1, -1)         # (1, 2048) for sequential splitting
print(f"  Combined: {all_tokens.shape}  -> {all_tokens.numel()} tokens")
assert all_tokens.numel() >= N_TOKENS, f"Need {N_TOKENS} tokens, got {all_tokens.numel()}"

# Use exactly 2048 tokens split into batches
all_tokens_trimmed = all_tokens.view(-1)[:N_TOKENS]  # flat 2048 token ids
# Reshape into (N_TOKENS//128, 128) for batched forward
token_batches = all_tokens_trimmed.view(-1, 128)  # (16, 128)

# ── Collect pre-GELU activations and FFN outputs per block ────────────────────
print("\n=== Collecting pre-GELU activations and FFN outputs per block ===")

block_z_all = {i: [] for i in range(N_BLOCKS)}   # pre-GELU z
block_ffn_out = {i: [] for i in range(N_BLOCKS)} # FFN output y

hooks = []

def make_z_hook(block_idx):
    def hook_fn(module, input, output):
        block_z_all[block_idx].append(output.detach().cpu())
    return hook_fn

# For FFN output we hook the entire mlp module
def make_ffn_hook(block_idx):
    def hook_fn(module, input, output):
        block_ffn_out[block_idx].append(output.detach().cpu())
    return hook_fn

for i in range(N_BLOCKS):
    h1 = model.transformer.h[i].mlp.c_fc.register_forward_hook(make_z_hook(i))
    h2 = model.transformer.h[i].mlp.register_forward_hook(make_ffn_hook(i))
    hooks.extend([h1, h2])

with torch.no_grad():
    for batch_start in range(0, token_batches.shape[0], BATCH_SIZE):
        batch = token_batches[batch_start:batch_start + BATCH_SIZE]
        _ = model(batch)

for h in hooks:
    h.remove()

# Concatenate: (n_tokens, 3072)
for i in range(N_BLOCKS):
    z_list = block_z_all[i]
    z_cat = torch.cat(z_list, dim=0)                       # (n_batches, 128, 3072)
    block_z_all[i] = z_cat.view(-1, z_cat.shape[-1])       # (2048, 3072)
    y_list = block_ffn_out[i]
    y_cat = torch.cat(y_list, dim=0)                        # (n_batches, 128, 768)
    block_ffn_out[i] = y_cat.view(-1, y_cat.shape[-1])     # (2048, 768)
    print(f"  Block {i}: z={block_z_all[i].shape}, y={block_ffn_out[i].shape}")

# ── Compute GELU derivative patterns ─────────────────────────────────────────
print("\n=== Computing GELU derivative patterns ===")
block_d = {}
for i in range(N_BLOCKS):
    z = block_z_all[i]  # (2048, 3072)
    d = gelu_deriv(z)   # (2048, 3072)
    # tanh-approx GELU derivative range: min ~-0.13 (at x~-0.5), max ~1.13 (at x>>0)
    assert d.min().item() >= -0.15, f"Block {i}: d.min={d.min().item():.4f} out of range"
    assert d.max().item() <= 1.15, f"Block {i}: d.max={d.max().item():.4f} out of range"
    block_d[i] = d
    print(f"  Block {i}: d range=[{d.min().item():.4f}, {d.max().item():.4f}]")

# ── Binary pattern analysis ───────────────────────────────────────────────────
print("\n=== Binary pattern analysis ===")
binary_results = {}
for i in range(N_BLOCKS):
    d = block_d[i]  # (2048, 3072)
    binary = (d > 0.5).numpy().astype(np.uint8)  # (2048, 3072)

    # Count unique binary patterns
    # Pack bits row-wise for hashing
    unique_rows = set()
    for row in binary:
        unique_rows.add(row.tobytes())
    n_unique = len(unique_rows)

    # Pairwise Hamming distances on 512 random pairs
    rng = np.random.default_rng(42)
    idx = rng.choice(N_TOKENS, size=512, replace=False)
    sample = binary[idx].astype(np.float32)  # (512, 3072)
    # Hamming = number of differing bits / 3072
    # Vectorized: for all pairs in sample, compute pairwise differences
    # Use only 128 pairs to avoid O(512^2) memory
    pairs_a = rng.integers(0, 512, size=128)
    pairs_b = rng.integers(0, 512, size=128)
    hamming_fracs = (sample[pairs_a] != sample[pairs_b]).mean(axis=1)

    binary_results[i] = {
        "n_unique_patterns": n_unique,
        "n_total_tokens": N_TOKENS,
        "hamming_mean": float(hamming_fracs.mean()),
        "hamming_std": float(hamming_fracs.std()),
        "hamming_min": float(hamming_fracs.min()),
        "hamming_max": float(hamming_fracs.max()),
    }
    print(f"  Block {i:2d}: unique_patterns={n_unique}/{N_TOKENS}, "
          f"Hamming mean={hamming_fracs.mean():.4f} std={hamming_fracs.std():.4f}")

# ── K-means sweep ─────────────────────────────────────────────────────────────
print("\n=== K-means sweep ===")

kmeans_results = {}
kstar_per_block = {}

for i in range(N_BLOCKS):
    z = block_z_all[i]       # (2048, 3072)
    d = block_d[i]           # (2048, 3072)
    y = block_ffn_out[i]     # (2048, 768)

    # Get W_out and b_proj from the model
    # GPT-2 uses Conv1D: c_proj.weight is (3072, 768), forward is x @ W (not x @ W.T)
    W_out = model.transformer.h[i].mlp.c_proj.weight.detach().cpu()  # (3072, 768)
    b_out = model.transformer.h[i].mlp.c_proj.bias.detach().cpu()    # (768,)

    z_train = z[:N_TRAIN].numpy()
    z_hold = z[N_TRAIN:].numpy()
    d_train = d[:N_TRAIN]
    d_hold = d[N_TRAIN:]
    y_hold = y[N_TRAIN:]

    # k=1 null baseline
    z_c1 = torch.from_numpy(z_train.mean(axis=0, keepdims=True))  # (1, 3072)
    d_c1 = gelu_deriv(z_c1)  # (1, 3072)
    z_c1_t = z_c1.squeeze(0)
    d_c1_t = d_c1.squeeze(0)

    z_hold_t = torch.from_numpy(z_hold)  # (512, 3072)
    # y_approx for k=1: all tokens use same centroid
    gelu_zc1 = F.gelu(z_c1_t.unsqueeze(0), approximate='tanh').squeeze(0)  # (3072,)
    delta1 = z_hold_t - z_c1_t.unsqueeze(0)  # (512, 3072)
    linearized1 = gelu_zc1.unsqueeze(0) + d_c1_t.unsqueeze(0) * delta1  # (512, 3072)
    y_approx1 = linearized1 @ W_out + b_out.unsqueeze(0)  # (512, 768)
    err1 = (y_approx1 - y_hold).norm(dim=1) / (y_hold.norm(dim=1) + 1e-8)
    null_error = err1.mean().item()
    print(f"\n  Block {i:2d}: k=1 null error = {null_error:.4f}")
    assert null_error > RECON_ERROR_THRESH, (
        f"Block {i}: k=1 error {null_error:.4f} <= 5% — experiment not informative"
    )

    block_k_results = {"k1_null_error": null_error, "k_sweep": {}}
    kstar = None

    for k in K_VALUES:
        km = MiniBatchKMeans(n_clusters=k, n_init=5, max_iter=300,
                              batch_size=max(256, k * 4), random_state=42)
        km.fit(z_train)

        # Assign holdout to clusters
        labels_hold = km.predict(z_hold)

        # Per-cluster centroid z_c = mean(z_train in cluster)
        labels_train = km.labels_

        # Compute Jensen gap diagnostic
        jensen_gaps = []

        # Build centroid stats
        cluster_z = {}
        cluster_sizes_train = {}
        for c in range(k):
            mask_c = (labels_train == c)
            if mask_c.sum() > 0:
                cluster_z[c] = torch.from_numpy(z_train[mask_c].mean(axis=0))
                cluster_sizes_train[c] = int(mask_c.sum())
            else:
                cluster_z[c] = torch.from_numpy(km.cluster_centers_[c])
                cluster_sizes_train[c] = 0

        # Compute d_c = GELU'(z_c) via autograd for each cluster
        cluster_d = {}
        all_zc = torch.stack([cluster_z[c] for c in range(k)], dim=0)  # (k, 3072)
        all_dc = gelu_deriv(all_zc)  # (k, 3072) — CORRECT method

        # Jensen gap: compare to mean of d within cluster
        for c in range(k):
            mask_c = (labels_train == c)
            if mask_c.sum() > 0:
                d_mean_in_cluster = d_train[mask_c].mean(dim=0)  # (3072,)
                gap = (all_dc[c] - d_mean_in_cluster).abs().mean().item()
                jensen_gaps.append(gap)

        # Reconstruction error on holdout
        z_hold_t = torch.from_numpy(z_hold)
        y_hold_t = y_hold

        errors = torch.zeros(N_HOLDOUT)
        degenerate_clusters = 0
        small_clusters_k = []

        for c in range(k):
            mask_h = (labels_hold == c)
            count = mask_h.sum()
            if count == 0:
                continue
            if cluster_sizes_train[c] < 3:
                degenerate_clusters += 1
                small_clusters_k.append(c)

            z_c = cluster_z[c]    # (3072,)
            d_c = all_dc[c]       # (3072,)
            gelu_zc = F.gelu(z_c.unsqueeze(0), approximate='tanh').squeeze(0)  # (3072,)

            z_h = z_hold_t[mask_h]       # (count, 3072)
            delta = z_h - z_c.unsqueeze(0)    # (count, 3072)
            linearized = gelu_zc.unsqueeze(0) + d_c.unsqueeze(0) * delta  # (count, 3072)
            y_approx = linearized @ W_out + b_out.unsqueeze(0)  # (count, 768)
            y_true_h = y_hold_t[mask_h]
            err = (y_approx - y_true_h).norm(dim=1) / (y_true_h.norm(dim=1) + 1e-8)
            errors[mask_h] = err

        mean_err = errors.mean().item()
        p95_err = errors.quantile(0.95).item()
        mean_jensen = float(np.mean(jensen_gaps)) if jensen_gaps else 0.0
        flag_jensen = mean_jensen > 0.05

        block_k_results["k_sweep"][k] = {
            "mean_error": mean_err,
            "p95_error": p95_err,
            "degenerate_clusters": degenerate_clusters,
            "mean_jensen_gap": mean_jensen,
            "flag_jensen": flag_jensen,
        }

        if kstar is None and mean_err < RECON_ERROR_THRESH:
            kstar = k

        print(f"  Block {i:2d} k={k:3d}: mean_err={mean_err:.4f} p95={p95_err:.4f} "
              f"degen={degenerate_clusters} jensen={mean_jensen:.4f}"
              + (" [KSTAR]" if k == kstar else ""))

    kstar_per_block[i] = kstar if kstar is not None else ">256"
    block_k_results["kstar"] = kstar_per_block[i]
    kmeans_results[i] = block_k_results
    print(f"  Block {i:2d}: K* = {kstar_per_block[i]}")

# ── Determine outcome ─────────────────────────────────────────────────────────
print("\n=== Outcome ===")
kstar_numeric = [v for v in kstar_per_block.values() if isinstance(v, int)]
kstar_over256 = sum(1 for v in kstar_per_block.values() if v == ">256")
kstar_le64 = sum(1 for v in kstar_per_block.values() if isinstance(v, int) and v <= 64)

print(f"  K* <= 64: {kstar_le64}/12 blocks")
print(f"  K* > 256: {kstar_over256}/12 blocks")
print(f"  K* values: {list(kstar_per_block.values())}")

if kstar_le64 >= 6:
    outcome = "CONFIRMED"
    outcome_detail = f"K* <= 64 for {kstar_le64}/12 blocks — FFN has manageable piecewise-linear complexity"
elif kstar_over256 > 6:
    outcome = "REFUTED"
    outcome_detail = f"K* > 256 for {kstar_over256}/12 blocks — GELU patterns too continuous to cluster"
else:
    outcome = "INCONCLUSIVE"
    outcome_detail = f"K* varies: {kstar_le64}/12 blocks <= 64, {kstar_over256}/12 blocks > 256"

print(f"\n  Outcome: {outcome}")
print(f"  {outcome_detail}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    "metadata": {
        "experiment": "cycle_085_gelu_pattern_diversity",
        "model": "gpt2",
        "n_tokens": N_TOKENS,
        "n_train": N_TRAIN,
        "n_holdout": N_HOLDOUT,
        "k_values": K_VALUES,
        "recon_error_thresh": RECON_ERROR_THRESH,
        "gelu_variant": "tanh_approximate",
    },
    "outcome": outcome,
    "outcome_detail": outcome_detail,
    "kstar_per_block": {str(k): v for k, v in kstar_per_block.items()},
    "kstar_le64_count": kstar_le64,
    "kstar_over256_count": kstar_over256,
    "binary_analysis": {str(k): v for k, v in binary_results.items()},
    "kmeans_results": {str(k): v for k, v in kmeans_results.items()},
}

json_path = os.path.join(RESULTS_DIR, "cycle_085_gelu_pattern_diversity.json")
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────
print("\n=== Plotting ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) K* per block bar chart
ax = axes[0]
blocks = list(range(N_BLOCKS))
kstar_vals_plot = []
colors = []
for i in blocks:
    v = kstar_per_block[i]
    if isinstance(v, int):
        kstar_vals_plot.append(v)
        colors.append("steelblue" if v <= 64 else "darkorange")
    else:
        kstar_vals_plot.append(300)  # placeholder for >256
        colors.append("crimson")

bars = ax.bar(blocks, kstar_vals_plot, color=colors)
ax.axhline(y=64, color='green', linestyle='--', alpha=0.7, label='K*=64 threshold')
ax.axhline(y=256, color='red', linestyle='--', alpha=0.7, label='K*=256 threshold')
ax.set_xlabel("Block")
ax.set_ylabel("K*")
ax.set_title("K* per block (min k for <5% reconstruction error)")
ax.set_xticks(blocks)
ax.legend()

# Annotate bars
for b, v in zip(bars, kstar_vals_plot):
    label = ">256" if v == 300 else str(v)
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, label,
            ha='center', va='bottom', fontsize=8)

# (b) Reconstruction error vs k curves for all 12 blocks
ax2 = axes[1]
cmap = plt.cm.get_cmap("tab20", N_BLOCKS)
for i in range(N_BLOCKS):
    errs = []
    for k in K_VALUES:
        errs.append(kmeans_results[i]["k_sweep"][k]["mean_error"])
    ax2.plot(K_VALUES, errs, marker='o', label=f"B{i}", color=cmap(i), alpha=0.8)

ax2.axhline(y=RECON_ERROR_THRESH, color='black', linestyle='--', alpha=0.7, label='5% threshold')
ax2.set_xlabel("k")
ax2.set_ylabel("Mean relative reconstruction error")
ax2.set_title("Reconstruction error vs k — all 12 blocks")
ax2.set_xscale("log")
ax2.legend(loc='upper right', fontsize=7, ncol=2)

fig.suptitle(f"GPT-2 FFN GELU Pattern Diversity — Outcome: {outcome}", fontsize=12)
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "cycle_085_gelu_pattern_diversity.png")
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
print(f"Saved plot: {plot_path}")

print("\n=== Done ===")
print(f"Outcome: {outcome}")
print(f"K* per block: {list(kstar_per_block.values())}")
