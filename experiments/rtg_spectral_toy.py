"""
Cycle 013: RTG Spectral Coarsening on Toy ReLU Network
-------------------------------------------------------
Builds Hamming-1 adjacency graph of observed activation patterns,
runs K-way spectral clustering, merges patterns via support-weighted
affine map averaging, and compares to random baseline.
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# ── Reproducibility ─────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")

# ── Dataset: spirals ─────────────────────────────────────────────────────────
def make_spirals(n=2000, turns=3, scale=2.0, noise=0.0, seed=42):
    rng = np.random.default_rng(seed)
    n_half = n // 2
    theta = np.linspace(0, turns * 2 * np.pi, n_half)
    r = np.linspace(0.1, 1.0, n_half)
    x0 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x1 = np.stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)], axis=1)
    X = np.concatenate([x0, x1], axis=0) * scale
    y = np.concatenate([np.zeros(n_half), np.ones(n_half)])
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X_all, y_all = make_spirals(n=2000, turns=3, scale=2.0, noise=0.0, seed=42)
n_train = 1000
X_train, y_train = X_all[:n_train], y_all[:n_train]
X_test, y_test = X_all[n_train:], y_all[n_train:]

# ── Network ──────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(h2).squeeze(-1)

    def forward_hidden(self, x):
        """Return (pre_relu1, post_relu1, pre_relu2, post_relu2, output)."""
        z1 = self.fc1(x)
        h1 = torch.relu(z1)
        z2 = self.fc2(h1)
        h2 = torch.relu(z2)
        out = self.fc3(h2).squeeze(-1)
        return z1, h1, z2, h2, out

# ── Training ─────────────────────────────────────────────────────────────────
model = MLP().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

print("Training 2->32->32->1 spirals network...")
best_acc = 0.0
for epoch in range(3000):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            train_acc = ((model(X_train) > 0) == y_train.bool()).float().mean().item()
            test_acc = ((model(X_test) > 0) == y_test.bool()).float().mean().item()
        print(f"  Epoch {epoch+1}: loss={loss.item():.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
        if train_acc >= 0.99:
            print(f"  Early stop at epoch {epoch+1}")
            break

model.eval()
with torch.no_grad():
    float_logits = model(X_test)
    float_preds = (float_logits > 0)
    float_acc = (float_preds == y_test.bool()).float().mean().item()
print(f"Float network test accuracy: {float_acc:.4f}")
assert float_acc >= 0.90, f"Network did not learn: acc={float_acc}"

# ── Extract activation patterns ──────────────────────────────────────────────
print("\nExtracting activation patterns on test set...")
model.eval()
with torch.no_grad():
    z1, h1, z2, h2, out = model.forward_hidden(X_test)

# Binary masks: 1 if h > 0
q1 = (h1 > 0).int().numpy()  # (1000, 32)
q2 = (h2 > 0).int().numpy()  # (1000, 32)
patterns_raw = np.concatenate([q1, q2], axis=1)  # (1000, 64)

# Find unique patterns
unique_patterns, inverse_idx, support_counts = np.unique(
    patterns_raw, axis=0, return_inverse=True, return_counts=True
)
N_patterns = unique_patterns.shape[0]
print(f"  Unique activation patterns: {N_patterns}")
print(f"  Support counts: min={support_counts.min()}, max={support_counts.max()}, mean={support_counts.mean():.1f}")

# ── Compute affine maps per pattern ──────────────────────────────────────────
print("\nComputing affine maps per pattern (MASO formula)...")
W1 = model.fc1.weight.detach().numpy()   # (32, 2)
b1 = model.fc1.bias.detach().numpy()     # (32,)
W2 = model.fc2.weight.detach().numpy()   # (32, 32)
b2 = model.fc2.bias.detach().numpy()     # (32,)
W3 = model.fc3.weight.detach().numpy()   # (1, 32)
b3 = model.fc3.bias.detach().numpy()     # (1,)

A_maps = np.zeros((N_patterns, 2))   # affine map A_ω: R^2 -> R^1 (slope)
b_maps = np.zeros(N_patterns)        # affine map b_ω (bias)

for i in range(N_patterns):
    q1_i = unique_patterns[i, :32].astype(float)  # (32,)
    q2_i = unique_patterns[i, 32:].astype(float)  # (32,)

    # A_ω = W3 @ diag(q2) @ W2 @ diag(q1) @ W1  shape: (1,2)
    M = W3 @ np.diag(q2_i) @ W2 @ np.diag(q1_i) @ W1  # (1, 2)
    A_maps[i] = M[0]  # (2,)

    # b_ω = W3 @ diag(q2) @ (W2 @ diag(q1) @ b1 + b2) + b3  scalar
    inner = W2 @ (q1_i * b1) + b2  # (32,)
    b_maps[i] = (W3 @ (q2_i * inner) + b3)[0]

# Sanity: reconstruct outputs using per-pattern affine maps, check vs float
print("Sanity check: K=N_patterns accuracy...")
reconstructed = np.zeros(len(X_test))
for i, pat_idx in enumerate(inverse_idx):
    x = X_test[i].numpy()
    reconstructed[i] = A_maps[pat_idx] @ x + b_maps[pat_idx]

recon_preds = reconstructed > 0
float_preds_np = float_preds.numpy()
sanity_acc = np.mean(recon_preds == float_preds_np)
print(f"  Affine reconstruction acc (K=N_patterns): {sanity_acc:.4f}")
if sanity_acc < 0.98:
    print("  WARNING: sanity check failed — check affine map computation")

# ── Build RTG (Hamming-1 adjacency) ──────────────────────────────────────────
print("\nBuilding RTG (Hamming-1 adjacency)...")
t0 = time.time()

# Use float patterns for cdist hamming
up_float = unique_patterns.astype(float)
# hamming distance * 64 = number of differing bits
ham_mat = cdist(up_float, up_float, 'hamming') * 64
adj_dense = (ham_mat == 1.0)
np.fill_diagonal(adj_dense, False)
adj_sparse = sp.csr_matrix(adj_dense.astype(float))

n_edges = adj_sparse.nnz // 2
degrees = np.asarray(adj_sparse.sum(axis=1)).flatten()
isolated = int(np.sum(degrees == 0))
n_components = sp.csgraph.connected_components(adj_sparse, directed=False)[0]
print(f"  RTG: {N_patterns} nodes, {n_edges} edges, {n_components} connected components")
print(f"  Degree: min={degrees.min():.0f}, max={degrees.max():.0f}, mean={degrees.mean():.1f}")
print(f"  Isolated nodes (degree=0): {isolated} ({100*isolated/N_patterns:.1f}%)")
print(f"  Build time: {time.time()-t0:.2f}s")

if isolated / N_patterns > 0.8:
    print("  WARNING: >80% isolated nodes, RTG is very sparse. Will try Hamming-2 fallback.")
    ham_mat2 = ham_mat.copy()
    adj_dense = (ham_mat2 <= 2.0)
    np.fill_diagonal(adj_dense, False)
    adj_sparse = sp.csr_matrix(adj_dense.astype(float))
    n_edges = adj_sparse.nnz // 2
    degrees = np.asarray(adj_sparse.sum(axis=1)).flatten()
    isolated = int(np.sum(degrees == 0))
    n_components = sp.csgraph.connected_components(adj_sparse, directed=False)[0]
    print(f"  Hamming-2 RTG: {N_patterns} nodes, {n_edges} edges, {n_components} components, {isolated} isolated")
    hamming_threshold = 2
else:
    hamming_threshold = 1

# ── Spectral clustering ───────────────────────────────────────────────────────
print("\nBuilding Laplacian and computing eigenvectors...")
D = sp.diags(degrees)
L = D - adj_sparse  # unnormalized Laplacian

K_values = [2, 4, 8, 16]
max_K = max(K_values)

# Compute enough eigenvectors for max_K
# Use sigma=0 (shift-invert) for smallest eigenvalues
try:
    eigenvalues, eigenvectors = spla.eigsh(L, k=max_K + 1, which='SM', tol=1e-10)
    print(f"  Computed {max_K+1} eigenvectors. Eigenvalues: {eigenvalues[:5]}")
    spectral_ok = True
except Exception as e:
    print(f"  eigsh failed: {e}")
    print("  Trying normalized Laplacian...")
    try:
        # Normalized Laplacian L_sym = D^{-1/2} L D^{-1/2}
        d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        L_sym = D_inv_sqrt @ L @ D_inv_sqrt
        eigenvalues, eigenvectors = spla.eigsh(L_sym, k=max_K + 1, which='SM', tol=1e-10)
        print(f"  Normalized Laplacian eigenvectors computed. Eigenvalues: {eigenvalues[:5]}")
        spectral_ok = True
    except Exception as e2:
        print(f"  Normalized Laplacian also failed: {e2}")
        spectral_ok = False

# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate_merged(cluster_labels, A_maps, b_maps, inverse_idx, support_counts,
                    X_test_np, float_preds_np):
    """Compute accuracy using support-weighted cluster-centroid affine maps."""
    K = int(cluster_labels.max()) + 1
    # Cluster centroids (support-weighted)
    A_cluster = np.zeros((K, 2))
    b_cluster = np.zeros(K)
    for k in range(K):
        members = np.where(cluster_labels == k)[0]
        weights = support_counts[members].astype(float)
        total_w = weights.sum()
        if total_w == 0:
            continue
        A_cluster[k] = (weights[:, None] * A_maps[members]).sum(axis=0) / total_w
        b_cluster[k] = (weights * b_maps[members]).sum() / total_w

    # Predict
    outputs = np.zeros(len(X_test_np))
    for i, pat_idx in enumerate(inverse_idx):
        k = cluster_labels[pat_idx]
        outputs[i] = A_cluster[k] @ X_test_np[i] + b_cluster[k]
    preds = outputs > 0
    acc = np.mean(preds == float_preds_np)
    return acc

X_test_np = X_test.numpy()

# ── Spectral results ──────────────────────────────────────────────────────────
spectral_accs = {}
if spectral_ok:
    print("\nSpectral clustering results:")
    for K in K_values:
        # Use first K eigenvectors (skip trivial eigenvector if needed)
        # eigsh returns in ascending order; first is trivial (all-same) for connected
        emb = eigenvectors[:, 1:K+1] if eigenvectors.shape[1] > K else eigenvectors[:, :K]
        # Normalize rows (standard spectral clustering)
        emb_norm = normalize(emb, norm='l2')
        # Handle all-zero rows (isolated nodes)
        zero_rows = np.all(emb_norm == 0, axis=1)
        if zero_rows.any():
            emb_norm[zero_rows] = 0.0  # will be assigned to cluster 0

        km = KMeans(n_clusters=K, n_init=20, random_state=42)
        labels = km.fit_predict(emb_norm)
        acc = evaluate_merged(labels, A_maps, b_maps, inverse_idx, support_counts,
                               X_test_np, float_preds_np)
        spectral_accs[K] = acc
        reduction = N_patterns / K
        print(f"  K={K:3d}: acc={acc:.4f}  reduction={reduction:.1f}x")

# ── Random baseline ───────────────────────────────────────────────────────────
print("\nRandom baseline:")
N_RANDOM_SEEDS = 10
random_accs = {}
random_stds = {}
for K in K_values:
    accs = []
    for seed in range(N_RANDOM_SEEDS):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, K, size=N_patterns)
        acc = evaluate_merged(labels, A_maps, b_maps, inverse_idx, support_counts,
                               X_test_np, float_preds_np)
        accs.append(acc)
    random_accs[K] = float(np.mean(accs))
    random_stds[K] = float(np.std(accs))
    print(f"  K={K:3d}: acc={random_accs[K]:.4f} ± {random_stds[K]:.4f}")

# ── Verdict ───────────────────────────────────────────────────────────────────
print("\n=== VERDICT ===")
print(f"Float accuracy:    {float_acc:.4f}")
print(f"Sanity (K=N):      {sanity_acc:.4f}")
print(f"N_patterns:        {N_patterns}")
print()
print(f"{'K':>4}  {'Spectral':>10}  {'Random':>10}  {'Delta':>8}  {'3x OK':>6}  {'<5% drop':>8}")
for K in K_values:
    if K in spectral_accs:
        spec = spectral_accs[K]
    else:
        spec = float('nan')
    rnd = random_accs[K]
    delta = spec - rnd if not np.isnan(spec) else float('nan')
    three_x_ok = "YES" if N_patterns / K >= 3 else "NO"
    drop = float_acc - spec if not np.isnan(spec) else float('nan')
    five_pct_ok = "YES" if drop < 0.05 else "NO"
    print(f"  {K:3d}  {spec:10.4f}  {rnd:10.4f}  {delta:+8.4f}  {three_x_ok:>6}  {five_pct_ok:>8}")

# Check agenda criteria
confirmed = False
refuted = False
inconclusive = False

if spectral_ok:
    # Check at any K: >= 3x reduction with < 5% drop AND >= 5pp over random
    for K in K_values:
        if K not in spectral_accs:
            continue
        spec = spectral_accs[K]
        rnd = random_accs[K]
        three_x = N_patterns / K >= 3
        small_drop = (float_acc - spec) < 0.05
        beats_random = (spec - rnd) >= 0.05
        if three_x and small_drop and beats_random:
            confirmed = True
            print(f"\nCONFIRMED at K={K}: 3x reduction={N_patterns/K:.1f}x, drop={float_acc-spec:.3f}, delta={spec-rnd:+.3f}")
            break
    if not confirmed:
        # Check if spectral never beats random at any K
        max_delta = max((spectral_accs.get(K, -999) - random_accs[K]) for K in K_values)
        if max_delta < 0.05:
            refuted = True
            print(f"\nREFUTED: spectral never beats random by >= 5pp (max delta = {max_delta:+.4f})")
        else:
            inconclusive = True
            print(f"\nINCONCLUSIVE: spectral beats random but criteria not fully met")
else:
    print("\nINCONCLUSIVE: spectral eigenvector computation failed")
    inconclusive = True

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
K_arr = np.array(K_values)
if spectral_ok:
    spec_arr = np.array([spectral_accs.get(K, np.nan) for K in K_values])
    ax.plot(K_arr, spec_arr, 'b-o', label='Spectral (RTG)', linewidth=2, markersize=8)
rnd_arr = np.array([random_accs[K] for K in K_values])
rnd_std_arr = np.array([random_stds[K] for K in K_values])
ax.fill_between(K_arr, rnd_arr - rnd_std_arr, rnd_arr + rnd_std_arr, alpha=0.3, color='orange')
ax.plot(K_arr, rnd_arr, 'r--o', label='Random baseline', linewidth=2, markersize=8)
ax.axhline(float_acc, color='green', linestyle=':', label=f'Float ({float_acc:.3f})', linewidth=2)
ax.axhline(float_acc - 0.05, color='gray', linestyle=':', label='Float - 5%', linewidth=1)
ax.set_xlabel('K (number of clusters)')
ax.set_ylabel('Accuracy (vs float preds)')
ax.set_title('RTG Spectral Coarsening vs Random Baseline')
ax.legend()
ax.set_xscale('log', base=2)
ax.set_xticks(K_values)
ax.set_xticklabels(K_values)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

ax2 = axes[1]
if spectral_ok and not np.all(np.isnan(spec_arr)):
    delta_arr = spec_arr - rnd_arr
    ax2.bar(K_arr, delta_arr, color=['green' if d >= 0.05 else 'red' for d in delta_arr])
    ax2.axhline(0.05, color='black', linestyle='--', label='5pp threshold')
    ax2.set_xlabel('K (number of clusters)')
    ax2.set_ylabel('Spectral - Random accuracy')
    ax2.set_title('Accuracy advantage of spectral over random')
    ax2.set_xticks(K_values)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'Spectral failed', ha='center', va='center', transform=ax2.transAxes)

plt.suptitle(f'RTG Spectral Coarsening — {N_patterns} patterns, {n_edges} edges\n'
             f'Hamming-{hamming_threshold} adjacency, {n_components} connected components',
             fontsize=12)
plt.tight_layout()
plot_path = 'autoresearch/results/cycle_013_rtg_spectral.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {plot_path}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
results = {
    "float_acc": float_acc,
    "sanity_acc_k_equals_n": sanity_acc,
    "N_patterns": int(N_patterns),
    "n_edges": int(n_edges),
    "n_connected_components": int(n_components),
    "n_isolated_nodes": int(isolated),
    "isolated_fraction": float(isolated) / N_patterns,
    "mean_degree": float(degrees.mean()),
    "hamming_threshold": int(hamming_threshold),
    "spectral_ok": bool(spectral_ok),
    "spectral_accs": {str(K): float(v) for K, v in spectral_accs.items()},
    "random_accs": {str(K): float(v) for K, v in random_accs.items()},
    "random_stds": {str(K): float(v) for K, v in random_stds.items()},
    "verdict": "CONFIRMED" if confirmed else ("REFUTED" if refuted else "INCONCLUSIVE"),
}
json_path = 'autoresearch/results/cycle_013_rtg_spectral.json'
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {json_path}")
print(f"\nVerdict: {results['verdict']}")
