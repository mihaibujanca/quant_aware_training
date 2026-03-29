"""
Cycle 101 — GELU output clustering (corrected) — GPT-2 FFN (item 106).

Supersedes cycle 085 (item 97) which used Taylor linearization (WRONG methodology).

Correct methodology:
  - Cluster tokens by GELU'(z) similarity (k-means on 3072-dim derivative vectors)
  - For each cluster, compute y_c = mean(actual FFN outputs) on training tokens
  - Reconstruction error on holdout: ||y_j - y_c|| / ||y_j||  (NO linearization)
  - K* = min k such that mean within-cluster reconstruction error < 5%
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
N_TOKENS = 4096   # 8 sequences per source × 4 sources × 128 length
N_TRAIN = 3072
N_HOLDOUT = 1024
K_VALUES = [8, 16, 32, 64, 128, 256]
RECON_ERROR_THRESH = 0.05  # 5%
BATCH_SIZE = 8
SEQS_PER_SOURCE = 8

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

# ── Build 4096-token probe corpus (8 sequences per source) ───────────────────
print("\n=== Building probe corpus ===")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
corpus_parts = []
for fname in ["shakespeare.txt", "wikitext2_valid.txt", "openwebtext_sample.txt", "c4_sample.txt"]:
    spec = ProbeCorpusSpec(
        text_path=os.path.join(DATA, fname),
        probe_train_sequences=SEQS_PER_SOURCE,
        sequence_length=128,
        teacher=teacher,
    )
    part = build_probe_corpus(spec)
    corpus_parts.append(part.probe_train)
    print(f"  {fname}: {part.probe_train.shape}")

all_tokens = torch.cat(corpus_parts, dim=0)  # (32, 128)
print(f"  Combined: {all_tokens.shape} -> {all_tokens.numel()} tokens")
assert all_tokens.numel() >= N_TOKENS, f"Need {N_TOKENS} tokens, got {all_tokens.numel()}"

# Use exactly N_TOKENS tokens split into batches of 128
all_tokens_trimmed = all_tokens.view(-1)[:N_TOKENS]
token_batches = all_tokens_trimmed.view(-1, 128)  # (32, 128)

# ── Collect pre-GELU activations and FFN outputs per block ────────────────────
print("\n=== Collecting pre-GELU activations and FFN outputs per block ===")

block_z_all = {i: [] for i in range(N_BLOCKS)}    # pre-GELU z (c_fc output)
block_ffn_out = {i: [] for i in range(N_BLOCKS)}  # full MLP output y

hooks = []

def make_z_hook(block_idx):
    def hook_fn(module, input, output):
        block_z_all[block_idx].append(output.detach().cpu())
    return hook_fn

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

# Concatenate and flatten: (N_TOKENS, dim)
for i in range(N_BLOCKS):
    z_cat = torch.cat(block_z_all[i], dim=0)
    block_z_all[i] = z_cat.view(-1, z_cat.shape[-1])   # (N_TOKENS, 3072)
    y_cat = torch.cat(block_ffn_out[i], dim=0)
    block_ffn_out[i] = y_cat.view(-1, y_cat.shape[-1]) # (N_TOKENS, 768)
    print(f"  Block {i}: z={block_z_all[i].shape}, y={block_ffn_out[i].shape}")

# ── Verify: hooked FFN output matches gelu(z) @ W_out + b_out ────────────────
print("\n=== Verifying hooked outputs match manual gelu(z)@W_out+b_out ===")
for i in [0, 5, 11]:
    W_out = model.transformer.h[i].mlp.c_proj.weight.detach().cpu()  # (3072, 768)
    b_out = model.transformer.h[i].mlp.c_proj.bias.detach().cpu()    # (768,)
    z_test = block_z_all[i][:32]   # first 32 tokens
    y_test = block_ffn_out[i][:32]
    y_manual = F.gelu(z_test, approximate='tanh') @ W_out + b_out
    max_err = (y_manual - y_test).abs().max().item()
    print(f"  Block {i}: max |y_manual - y_hook| = {max_err:.2e}")
    assert max_err < 1e-4, f"Block {i}: large discrepancy {max_err:.2e} — check MLP structure"

print("  Verification passed: hooked output matches gelu(z)@W_out+b_out")

# ── Compute GELU derivative patterns (clustering features) ───────────────────
print("\n=== Computing GELU derivative patterns ===")
block_d = {}
for i in range(N_BLOCKS):
    z = block_z_all[i]
    d = gelu_deriv(z)
    assert d.min().item() >= -0.15, f"Block {i}: d.min={d.min().item():.4f} out of range"
    assert d.max().item() <= 1.15, f"Block {i}: d.max={d.max().item():.4f} out of range"
    block_d[i] = d
    print(f"  Block {i}: d range=[{d.min().item():.4f}, {d.max().item():.4f}]")

# ── K-means sweep with CORRECT reconstruction (cluster-mean of actual outputs) ──
print("\n=== K-means sweep (correct methodology) ===")

kmeans_results = {}
kstar_per_block = {}

for i in range(N_BLOCKS):
    z = block_z_all[i]       # (N_TOKENS, 3072)
    d = block_d[i]           # (N_TOKENS, 3072) — clustering features
    y = block_ffn_out[i]     # (N_TOKENS, 768) — target outputs

    # Train/holdout split
    d_train = d[:N_TRAIN].numpy()   # cluster features for k-means
    d_hold = d[N_TRAIN:].numpy()
    y_train = y[:N_TRAIN]           # actual FFN outputs for computing cluster means
    y_hold = y[N_TRAIN:]            # holdout outputs for error evaluation

    # k=1 null baseline: single global mean output
    y_c1 = y_train.mean(dim=0)  # (768,) — global mean output
    err1 = (y_hold - y_c1.unsqueeze(0)).norm(dim=1) / (y_hold.norm(dim=1) + 1e-8)
    null_error = err1.mean().item()
    print(f"\n  Block {i:2d}: k=1 null error (correct) = {null_error:.4f}")
    assert null_error > RECON_ERROR_THRESH, (
        f"Block {i}: k=1 error {null_error:.4f} <= 5% — experiment not informative"
    )

    block_k_results = {"k1_null_error": null_error, "k_sweep": {}}
    kstar = None
    prev_mean_err = null_error

    for k in K_VALUES:
        # Cluster on GELU'(z) features (3072-dim derivative vectors)
        km = MiniBatchKMeans(n_clusters=k, n_init=5, max_iter=300,
                             batch_size=max(256, k * 4), random_state=42)
        km.fit(d_train)

        labels_train = km.labels_
        labels_hold = km.predict(d_hold)

        # Compute cluster mean outputs from TRAINING tokens (actual FFN outputs)
        cluster_mean_y = torch.zeros(k, y_train.shape[1])  # (k, 768)
        cluster_sizes_train = torch.zeros(k, dtype=torch.long)
        for c in range(k):
            mask_c = (labels_train == c)
            if mask_c.sum() > 0:
                cluster_mean_y[c] = y_train[mask_c].mean(dim=0)
                cluster_sizes_train[c] = int(mask_c.sum())
            # If empty train cluster, y_c stays zero — will be flagged as degenerate

        # Compute per-holdout-token reconstruction error
        degenerate_clusters = int((cluster_sizes_train < 3).sum().item())

        # Gather cluster mean output for each holdout token
        labels_hold_t = torch.from_numpy(labels_hold).long()
        y_approx = cluster_mean_y[labels_hold_t]  # (N_HOLDOUT, 768)

        errors = (y_approx - y_hold).norm(dim=1) / (y_hold.norm(dim=1) + 1e-8)
        mean_err = errors.mean().item()
        p95_err = errors.quantile(0.95).item()

        # Within-cluster output variance: mean over clusters of var(y_i in c)
        within_var_list = []
        for c in range(k):
            mask_h = (labels_hold_t == c)
            if mask_h.sum() > 1:
                within_var_list.append(y_hold[mask_h].var(dim=0).mean().item())
        within_var = float(np.mean(within_var_list)) if within_var_list else 0.0

        # Between-cluster output variance: var of cluster means
        # Only consider clusters with holdout members
        active_clusters = list(set(labels_hold))
        if len(active_clusters) > 1:
            cluster_means_active = cluster_mean_y[active_clusters]
            between_var = cluster_means_active.var(dim=0).mean().item()
        else:
            between_var = 0.0

        var_ratio = between_var / (within_var + 1e-10)

        # Monotonicity check
        monotone_ok = (mean_err <= prev_mean_err * 1.05)  # allow 5% slack
        prev_mean_err = mean_err

        block_k_results["k_sweep"][k] = {
            "mean_error": mean_err,
            "p95_error": p95_err,
            "within_cluster_var": within_var,
            "between_cluster_var": between_var,
            "variance_ratio": var_ratio,
            "degenerate_clusters": degenerate_clusters,
            "monotone_ok": monotone_ok,
        }

        if kstar is None and mean_err < RECON_ERROR_THRESH:
            kstar = k

        print(f"  Block {i:2d} k={k:3d}: mean_err={mean_err:.4f} p95={p95_err:.4f} "
              f"var_ratio={var_ratio:.2f} degen={degenerate_clusters}"
              + (" [KSTAR]" if k == kstar else "")
              + ("" if monotone_ok else " [NON-MONOTONE]"))

    kstar_per_block[i] = kstar if kstar is not None else ">256"
    block_k_results["kstar"] = kstar_per_block[i]
    kmeans_results[i] = block_k_results
    print(f"  Block {i:2d}: K* = {kstar_per_block[i]}")

# ── Determine outcome ─────────────────────────────────────────────────────────
print("\n=== Outcome ===")
kstar_numeric = [v for v in kstar_per_block.values() if isinstance(v, int)]
kstar_over256 = sum(1 for v in kstar_per_block.values() if v == ">256")
kstar_le64 = sum(1 for v in kstar_per_block.values() if isinstance(v, int) and v <= 64)

print(f"  K* <= 64:  {kstar_le64}/12 blocks")
print(f"  K* > 256:  {kstar_over256}/12 blocks")
print(f"  K* values: {list(kstar_per_block.values())}")

if kstar_le64 >= 6:
    outcome = "CONFIRMED"
    outcome_detail = (f"K* <= 64 for {kstar_le64}/12 blocks — "
                      "tokens in same GELU' cluster produce similar actual outputs")
elif kstar_over256 > 6:
    outcome = "REFUTED"
    outcome_detail = (f"K* > 256 for {kstar_over256}/12 blocks — "
                      "GELU' cluster membership does not predict output similarity")
else:
    outcome = "INCONCLUSIVE"
    outcome_detail = (f"K* varies: {kstar_le64}/12 blocks <= 64, "
                      f"{kstar_over256}/12 blocks > 256")

print(f"\n  Outcome: {outcome}")
print(f"  {outcome_detail}")

# ── Cross-check: compare against old cycle 085 values ────────────────────────
# The correct method (cluster-mean of actual outputs) should give LOWER error
# than the wrong Taylor linearization. Log k=256 errors for reference.
print("\n=== k=256 errors (for cross-check with cycle 085) ===")
for i in range(N_BLOCKS):
    err_256 = kmeans_results[i]["k_sweep"].get(256, {}).get("mean_error", float('nan'))
    print(f"  Block {i:2d}: k=256 error = {err_256:.4f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    "metadata": {
        "experiment": "cycle_101_gelu_output_clustering_corrected",
        "supersedes": "cycle_085_item_97",
        "model": "gpt2",
        "n_tokens": N_TOKENS,
        "n_train": N_TRAIN,
        "n_holdout": N_HOLDOUT,
        "k_values": K_VALUES,
        "recon_error_thresh": RECON_ERROR_THRESH,
        "gelu_variant": "tanh_approximate",
        "clustering_feature": "gelu_derivative",
        "reconstruction_method": "cluster_mean_actual_output",
    },
    "outcome": outcome,
    "outcome_detail": outcome_detail,
    "kstar_per_block": {str(k): v for k, v in kstar_per_block.items()},
    "kstar_le64_count": kstar_le64,
    "kstar_over256_count": kstar_over256,
    "kmeans_results": {str(k): v for k, v in kmeans_results.items()},
}

json_path = os.path.join(RESULTS_DIR, "cycle_101_gelu_output_clustering.json")
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ── Plot: heatmap of mean reconstruction error (blocks x k) ─────────────────
print("\n=== Plotting ===")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# (a) Heatmap: blocks (rows) × k (columns)
ax = axes[0]
error_matrix = np.zeros((N_BLOCKS, len(K_VALUES)))
for i in range(N_BLOCKS):
    for j, k in enumerate(K_VALUES):
        error_matrix[i, j] = kmeans_results[i]["k_sweep"][k]["mean_error"]

im = ax.imshow(error_matrix, aspect='auto', cmap='RdYlGn_r',
               vmin=0.0, vmax=max(0.1, error_matrix.max()))
ax.set_xticks(range(len(K_VALUES)))
ax.set_xticklabels([str(k) for k in K_VALUES])
ax.set_yticks(range(N_BLOCKS))
ax.set_yticklabels([f"B{i}" for i in range(N_BLOCKS)])
ax.set_xlabel("k (number of clusters)")
ax.set_ylabel("Block")
ax.set_title("Mean reconstruction error\n(cluster-mean of actual outputs)")
plt.colorbar(im, ax=ax, label="Relative error")

# Annotate cells with values and mark where < 5%
for i in range(N_BLOCKS):
    for j, k in enumerate(K_VALUES):
        val = error_matrix[i, j]
        color = 'white' if val > 0.3 else 'black'
        marker = '*' if val < RECON_ERROR_THRESH else ''
        ax.text(j, i, f"{val:.2f}{marker}", ha='center', va='center',
                fontsize=6, color=color)

# (b) K* per block bar chart
ax2 = axes[1]
kstar_vals_plot = []
colors = []
for i in range(N_BLOCKS):
    v = kstar_per_block[i]
    if isinstance(v, int):
        kstar_vals_plot.append(v)
        colors.append("steelblue" if v <= 64 else "darkorange")
    else:
        kstar_vals_plot.append(300)
        colors.append("crimson")

blocks = list(range(N_BLOCKS))
bars = ax2.bar(blocks, kstar_vals_plot, color=colors)
ax2.axhline(y=64, color='green', linestyle='--', alpha=0.7, label='K*=64 threshold')
ax2.axhline(y=256, color='red', linestyle='--', alpha=0.7, label='K*=256 threshold')
ax2.set_xlabel("Block")
ax2.set_ylabel("K*")
ax2.set_title("K* per block\n(min k for <5% reconstruction error)")
ax2.set_xticks(blocks)
ax2.legend()

for b, v in zip(bars, kstar_vals_plot):
    label = ">256" if v == 300 else str(v)
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 3, label,
             ha='center', va='bottom', fontsize=8)

fig.suptitle(
    f"GPT-2 FFN GELU Output Clustering (corrected) — Outcome: {outcome}\n"
    f"Clustering: GELU'(z), Reconstruction: cluster-mean actual outputs",
    fontsize=11
)
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "cycle_101_gelu_output_clustering.png")
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
print(f"Saved plot: {plot_path}")

print("\n=== Done ===")
print(f"Outcome: {outcome}")
print(f"K* per block: {list(kstar_per_block.values())}")
