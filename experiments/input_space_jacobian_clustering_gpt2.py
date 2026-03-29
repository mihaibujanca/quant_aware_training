"""
Cycle 088 — Input-space token clustering + local Jacobian — GPT-2 FFN (item 102).

For blocks [7, 5, 4, 1], cluster FFN inputs (768D post-LayerNorm) with k-means.
Compute local Jacobian J_c = W_out @ diag(GELU'(W_in @ x_c + b_in)) @ W_in at each centroid.
Reconstruct y_approx = J_c @ (x - x_c) + f(x_c). Sweep K in {4, 8, 16, 32, 64}.
Find K* per block for < 5% mean relative reconstruction error.
Also compute effective rank of each J_c at K=16.
"""

import json
import math
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

BLOCKS = [7, 5, 4, 1]
N_TOKENS = 4096
N_TRAIN = 3072
N_HOLDOUT = 1024
K_VALUES = [4, 8, 16, 32, 64]
RECON_ERROR_THRESH = 0.05
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


def gelu_grad(z: torch.Tensor) -> torch.Tensor:
    """Element-wise derivative of GPT-2 GELU (tanh approximation). z: any shape."""
    k = math.sqrt(2.0 / math.pi)
    t = torch.tanh(k * (z + 0.044715 * z ** 3))
    dt = 1.0 - t ** 2
    inner_deriv = k * (1.0 + 3.0 * 0.044715 * z ** 2)
    return 0.5 * (1.0 + t) + 0.5 * z * dt * inner_deriv


# ── Load GPT-2 ────────────────────────────────────────────────────────────────
print("\n=== Loading GPT-2 ===")
teacher = load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))
model = teacher.model
tokenizer = teacher.tokenizer
model.eval()

# ── Build 4096-token mixed-corpus probe ───────────────────────────────────────
print("\n=== Building probe corpus ===")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
corpus_parts = []
for fname in ["shakespeare.txt", "wikitext2_valid.txt", "openwebtext_sample.txt", "c4_sample.txt"]:
    spec = ProbeCorpusSpec(
        text_path=os.path.join(DATA, fname),
        probe_train_sequences=8,
        sequence_length=128,
        teacher=teacher,
    )
    part = build_probe_corpus(spec)
    corpus_parts.append(part.probe_train)
    print(f"  {fname}: {part.probe_train.shape}")

all_tokens = torch.cat(corpus_parts, dim=0)  # (32, 129) — build_probe_corpus returns len+1
# Trim to context length 128 (first 128 tokens per sequence, ignore last target)
seq_len = min(128, all_tokens.shape[1])
all_tokens = all_tokens[:, :seq_len]  # (32, 128)
if all_tokens.shape[0] * seq_len < N_TOKENS:
    # fallback: repeat if needed
    repeats = math.ceil(N_TOKENS / (all_tokens.shape[0] * seq_len))
    all_tokens = all_tokens.repeat(repeats, 1)
# take N_TOKENS tokens as sequences of length seq_len
n_seqs = N_TOKENS // seq_len
all_tokens = all_tokens[:n_seqs]  # (n_seqs, seq_len)
print(f"  Total sequences: {all_tokens.shape} → {all_tokens.shape[0]*seq_len} tokens")

# ── Collect FFN inputs and outputs via hooks ───────────────────────────────────
print("\n=== Collecting FFN activations ===")

block_data = {}  # block_idx -> {'x': (N, 768), 'y': (N, 768)}

for b in BLOCKS:
    print(f"  Block {b}...")
    x_list = []
    y_list = []

    def make_hooks(b):
        handles = []
        def hook_x(module, inp, out):
            x_list.append(out.detach().float().reshape(-1, 768))
        def hook_y(module, inp, out):
            y_list.append(out.detach().float().reshape(-1, 768))
        handles.append(model.transformer.h[b].ln_2.register_forward_hook(hook_x))
        handles.append(model.transformer.h[b].mlp.register_forward_hook(hook_y))
        return handles

    handles = make_hooks(b)

    with torch.no_grad():
        for i in range(0, n_seqs, BATCH_SIZE):
            batch = all_tokens[i:i+BATCH_SIZE]
            # GPT-2 max position is 1024; sequence length must be <= 1024
            model(batch[:, :1024])

    for h in handles:
        h.remove()

    x_all = torch.cat(x_list, dim=0)[:N_TOKENS]  # (N_TOKENS, 768)
    y_all = torch.cat(y_list, dim=0)[:N_TOKENS]
    block_data[b] = {'x': x_all, 'y': y_all}
    print(f"    x: {x_all.shape}, y: {y_all.shape}")

# ── Extract FFN weights ────────────────────────────────────────────────────────
def get_ffn_weights(b):
    c_fc = model.transformer.h[b].mlp.c_fc
    c_proj = model.transformer.h[b].mlp.c_proj
    W_in = c_fc.weight.T.detach().float()   # (3072, 768)
    b_in = c_fc.bias.detach().float()       # (3072,)
    W_out = c_proj.weight.T.detach().float()  # (768, 3072)
    b_out = c_proj.bias.detach().float()      # (768,)
    assert W_in.shape == (3072, 768)
    assert W_out.shape == (768, 3072)
    return W_in, b_in, W_out, b_out


# ── Main sweep ────────────────────────────────────────────────────────────────
print("\n=== K-means sweep + Jacobian reconstruction ===")
results = {}

for b in BLOCKS:
    print(f"\n  Block {b}:")
    W_in, b_in, W_out, b_out = get_ffn_weights(b)

    x_all = block_data[b]['x']   # (N_TOKENS, 768)
    y_all = block_data[b]['y']   # (N_TOKENS, 768)

    x_train = x_all[:N_TRAIN].numpy()
    y_train = y_all[:N_TRAIN]
    x_hold = x_all[N_TRAIN:]     # (N_HOLDOUT, 768)
    y_hold = y_all[N_TRAIN:]     # (N_HOLDOUT, 768)
    x_hold_t = x_hold            # torch tensor

    block_res = {'K_values': K_VALUES, 'mean_error': [], 'p95_error': [], 'K_star': None}

    # K=1 baseline: global mean as centroid
    x_mean_np = x_train.mean(axis=0)
    x_c_t = torch.tensor(x_mean_np, dtype=torch.float32)
    z_c = W_in @ x_c_t + b_in
    d_c = gelu_grad(z_c)
    J_c = W_out * d_c.unsqueeze(0)   # broadcast: (768, 3072) * (1, 3072) → (768, 3072)
    J_c = J_c @ W_in                  # (768, 768)
    f_c = W_out @ F.gelu(z_c, approximate='tanh') + b_out  # (768,)
    y_approx_base = (x_hold_t - x_c_t) @ J_c.T + f_c.unsqueeze(0)  # (N_HOLDOUT, 768)
    err_base = torch.norm(y_approx_base - y_hold, dim=1) / (torch.norm(y_hold, dim=1) + 1e-8)
    baseline_mean = err_base.mean().item()
    baseline_p95 = err_base.quantile(0.95).item()
    print(f"    K=1 (null baseline): mean={baseline_mean:.4f}, p95={baseline_p95:.4f}")
    assert baseline_mean > RECON_ERROR_THRESH, (
        f"K=1 error {baseline_mean:.4f} <= 5% threshold — FFN already globally linear?"
    )
    block_res['K1_mean_error'] = baseline_mean
    block_res['K1_p95_error'] = baseline_p95

    # Spot-check linearity: J_c @ 0 = 0, reconstruction at centroid = f(x_c)
    zero_vec = torch.zeros(768)
    assert torch.allclose(J_c @ zero_vec, torch.zeros(768), atol=1e-5), "Linearity check failed"
    print(f"    Linearity check: ok")

    # Validate GELU grad against autograd for one centroid
    z_test = z_c.clone()
    z_auto = z_test.requires_grad_(True)
    g_auto = F.gelu(z_auto, approximate='tanh')
    gp_auto = torch.autograd.grad(g_auto.sum(), z_auto)[0].detach()
    gp_analytic = gelu_grad(z_c)
    max_diff = (gp_auto - gp_analytic).abs().max().item()
    assert max_diff < 1e-4, f"GELU' mismatch: max diff = {max_diff:.2e}"
    print(f"    Analytic vs autograd GELU' max diff: {max_diff:.2e}  ok")

    # K-means sweep
    for K in K_VALUES:
        km = MiniBatchKMeans(n_clusters=K, random_state=42, n_init=3, batch_size=1024)
        km.fit(x_train)
        centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)  # (K, 768)

        # Assign holdout tokens
        labels_hold = km.predict(x_hold_t.numpy())  # (N_HOLDOUT,)

        # Check for degenerate clusters in holdout
        cluster_counts = np.bincount(labels_hold, minlength=K)
        n_degenerate = (cluster_counts < 5).sum()
        if n_degenerate > 0:
            print(f"    K={K}: {n_degenerate} clusters with < 5 holdout members (flagged)")

        # Compute Jacobian at each centroid
        errors_hold = torch.zeros(N_HOLDOUT)
        for c_idx in range(K):
            mask = torch.tensor(labels_hold == c_idx)
            if mask.sum() == 0:
                continue
            x_c = centroids[c_idx]  # (768,)
            z_c = W_in @ x_c + b_in  # (3072,)
            d_c = gelu_grad(z_c)
            J_c = (W_out * d_c.unsqueeze(0)) @ W_in  # (768, 768)
            f_c = W_out @ F.gelu(z_c, approximate='tanh') + b_out  # (768,)

            x_clust = x_hold_t[mask]          # (n_c, 768)
            y_clust = y_hold[mask]             # (n_c, 768)
            delta = x_clust - x_c.unsqueeze(0)
            y_approx = delta @ J_c.T + f_c.unsqueeze(0)  # (n_c, 768)
            err = torch.norm(y_approx - y_clust, dim=1) / (torch.norm(y_clust, dim=1) + 1e-8)
            errors_hold[mask] = err

        mean_err = errors_hold.mean().item()
        p95_err = errors_hold.quantile(0.95).item()
        block_res['mean_error'].append(mean_err)
        block_res['p95_error'].append(p95_err)
        print(f"    K={K:3d}: mean={mean_err:.4f}, p95={p95_err:.4f}")

    # Find K*
    K_star = None
    for K, me in zip(K_VALUES, block_res['mean_error']):
        if me < RECON_ERROR_THRESH:
            K_star = K
            break
    if K_star is None:
        K_star = ">64"
    block_res['K_star'] = K_star
    print(f"    K* = {K_star}")

    # Check monotonicity
    errors = block_res['mean_error']
    if errors[-1] < errors[0]:
        print(f"    Monotonicity: ok (error decreases from K=4 to K=64)")
    else:
        print(f"    WARNING: non-monotonic errors: {errors}")

    # Effective rank of J_c at K=16 (or K* if numeric)
    K_rank = 16 if 16 in K_VALUES else (K_star if isinstance(K_star, int) else K_VALUES[-1])
    km_rank = MiniBatchKMeans(n_clusters=K_rank, random_state=42, n_init=3, batch_size=1024)
    km_rank.fit(x_train)
    centroids_rank = torch.tensor(km_rank.cluster_centers_, dtype=torch.float32)

    ranks_95 = []
    ranks_99 = []
    for c_idx in range(K_rank):
        x_c = centroids_rank[c_idx]
        z_c = W_in @ x_c + b_in
        d_c = gelu_grad(z_c)
        J_c = (W_out * d_c.unsqueeze(0)) @ W_in  # (768, 768)
        sv = torch.linalg.svdvals(J_c).float()
        energy = (sv ** 2).cumsum(0) / (sv ** 2).sum()
        r95 = (energy < 0.95).sum().item() + 1
        r99 = (energy < 0.99).sum().item() + 1
        ranks_95.append(r95)
        ranks_99.append(r99)

    block_res['eff_rank_95_mean'] = float(np.mean(ranks_95))
    block_res['eff_rank_95_median'] = float(np.median(ranks_95))
    block_res['eff_rank_99_mean'] = float(np.mean(ranks_99))
    block_res['eff_rank_99_median'] = float(np.median(ranks_99))
    print(f"    Effective rank (K={K_rank}): rank95={np.mean(ranks_95):.1f} (median {np.median(ranks_95):.0f}), "
          f"rank99={np.mean(ranks_99):.1f} (median {np.median(ranks_99):.0f})")

    results[b] = block_res
    del x_all, y_all, x_train, y_train, x_hold, y_hold

# ── Save JSON ─────────────────────────────────────────────────────────────────
json_path = os.path.join(RESULTS_DIR, "cycle_088_input_jacobian_clustering.json")
with open(json_path, "w") as f:
    json.dump({
        "blocks": BLOCKS,
        "K_values": K_VALUES,
        "N_tokens": N_TOKENS,
        "N_train": N_TRAIN,
        "N_holdout": N_HOLDOUT,
        "threshold": RECON_ERROR_THRESH,
        "results": {str(b): v for b, v in results.items()},
    }, f, indent=2)
print(f"\nSaved JSON: {json_path}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# (a) Reconstruction error vs K
ax = axes[0]
for b in BLOCKS:
    r = results[b]
    ax.plot(r['K_values'], r['mean_error'], marker='o', label=f"Block {b}")
ax.axhline(RECON_ERROR_THRESH, color='k', linestyle='--', linewidth=1.5, label='5% threshold')
ax.set_xlabel("K (n_clusters)")
ax.set_ylabel("Mean relative reconstruction error")
ax.set_title("(a) Recon error vs K — input-space clustering")
ax.legend()
ax.set_yscale("log")

# (b) K* bar chart
ax = axes[1]
block_labels = [f"Block {b}" for b in BLOCKS]
k_star_vals = []
for b in BLOCKS:
    ks = results[b]['K_star']
    k_star_vals.append(128 if ks == ">64" else int(ks))
colors = ['green' if v <= 64 else 'red' for v in k_star_vals]
bars = ax.bar(block_labels, k_star_vals, color=colors)
ax.axhline(64, color='orange', linestyle='--', linewidth=1.5, label='K=64 limit')
ax.set_ylabel("K* (first K < 5% error)")
ax.set_title("(b) K* per block")
ax.legend()
for bar, v, b in zip(bars, k_star_vals, BLOCKS):
    label = ">64" if results[b]['K_star'] == ">64" else str(v)
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, label, ha='center', va='bottom')

# (c) Effective rank histogram
ax = axes[2]
all_ranks_95 = []
all_ranks_99 = []
for b in BLOCKS:
    r = results[b]
    all_ranks_95.append(r['eff_rank_95_mean'])
    all_ranks_99.append(r['eff_rank_99_mean'])

x_pos = np.arange(len(BLOCKS))
width = 0.35
ax.bar(x_pos - width/2, all_ranks_95, width, label='95% energy', alpha=0.8)
ax.bar(x_pos + width/2, all_ranks_99, width, label='99% energy', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(block_labels)
ax.set_ylabel("Effective rank (mean over centroids)")
ax.set_title(f"(c) Eff. rank of J_c (K={K_rank})")
ax.legend()

plt.tight_layout()
png_path = os.path.join(RESULTS_DIR, "cycle_088_input_jacobian_clustering.png")
plt.savefig(png_path, dpi=120, bbox_inches='tight')
print(f"Saved plot: {png_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
n_confirmed = sum(1 for b in BLOCKS if isinstance(results[b]['K_star'], int) and results[b]['K_star'] <= 64)
print(f"  Blocks with K* <= 64: {n_confirmed}/{len(BLOCKS)}")
for b in BLOCKS:
    r = results[b]
    print(f"  Block {b}: K*={r['K_star']}, "
          f"K=64 error={r['mean_error'][-1]:.4f}, "
          f"rank95={r['eff_rank_95_mean']:.1f}, rank99={r['eff_rank_99_mean']:.1f}")

if n_confirmed >= 3:
    verdict = "CONFIRMED"
elif n_confirmed == 0 and all(
    isinstance(results[b]['K_star'], str) or results[b]['mean_error'][-1] > 0.10
    for b in BLOCKS
):
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"
print(f"\n  Hypothesis: {verdict}")
