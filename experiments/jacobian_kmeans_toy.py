"""
Cycle 012 — Effective number of affine behaviors via Jacobian k-means — toy ReLU.

Train a 2→32→32→1 spirals MLP, extract all observed activation patterns,
compute their exact MASO affine maps A_ω, cluster with k-means on flattened
A_ω vectors, find minimum K* with ≤1% accuracy drop vs float.
"""

import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/mihai/Projects/quant_aware_training")

from aleph.datasets import make_spirals
from aleph.relu_geometric_compression import (
    _forward_hidden,
    _pack_activation_codes,
    _effective_affine_from_mask,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Dataset ──────────────────────────────────────────────────────────────────
X_np, y_np = make_spirals(n_samples=2000, noise=0.0, n_turns=3, random_state=SEED)
X_np = X_np * 2.0  # scale=2.0 per plan
y_np = y_np.astype(np.float32)

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

# 50/50 train/test split
n_train = 1000
perm = torch.randperm(2000, generator=torch.Generator().manual_seed(SEED))
train_idx, test_idx = perm[:n_train], perm[n_train:]
X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]


# ── Architecture ─────────────────────────────────────────────────────────────
class SimpleMLP(nn.Module):
    """2→32→32→1 ReLU MLP with .layers Sequential, compatible with _linear_layers()."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


model = SimpleMLP()
torch.manual_seed(SEED)
# re-init to get a deterministic init
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# ── Training ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(3000):
    model.train()
    logits = model(X_train).squeeze(1)
    loss = F.binary_cross_entropy_with_logits(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        model.eval()
        with torch.no_grad():
            tr_acc = ((model(X_train).squeeze(1) > 0).float() == y_train).float().mean().item()
            te_acc = ((model(X_test).squeeze(1) > 0).float() == y_test).float().mean().item()
        print(f"Epoch {epoch+1:4d}  loss={loss.item():.4f}  train_acc={tr_acc:.3f}  test_acc={te_acc:.3f}")
        if tr_acc >= 0.99:
            print("  → train_acc ≥ 99%, stopping early")
            break

model.eval()
with torch.no_grad():
    train_acc = ((model(X_train).squeeze(1) > 0).float() == y_train).float().mean().item()
    test_acc  = ((model(X_test).squeeze(1) > 0).float() == y_test).float().mean().item()

print(f"\nFinal: train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

if test_acc < 0.90:
    print("ERROR: test accuracy below 90% — model underfitting, aborting analysis.")
    sys.exit(1)

# ── Extract activation patterns on test set ──────────────────────────────────
with torch.no_grad():
    logits_test, masks_list, _, _ = _forward_hidden(model, X_test)

# masks_list: list of (1000, 32) tensors, one per hidden layer
flat_masks, code_keys = _pack_activation_codes(masks_list)
# flat_masks: (1000, 64) bool numpy array
# code_keys: tuple of 1000 hex strings

unique_keys = list(dict.fromkeys(code_keys))  # preserves first-seen order
N_patterns = len(unique_keys)
print(f"\nUnique activation patterns (test set): N_patterns = {N_patterns}")

# Per-unique-pattern: compute the exact affine map A_ω (shape 2,), b_ω (scalar)
# Build mapping key -> index in unique_keys
key_to_uid = {k: i for i, k in enumerate(unique_keys)}

# For each unique pattern, compute A_ω and b_ω
affine_vecs = np.zeros((N_patterns, 3), dtype=np.float32)  # [A_w[0], A_w[1], b_w]

# We need to find one representative sample index per unique key
key_to_sample_idx = {}
for i, k in enumerate(code_keys):
    if k not in key_to_sample_idx:
        key_to_sample_idx[k] = i

with torch.no_grad():
    for uid, key in enumerate(unique_keys):
        sample_idx = key_to_sample_idx[key]
        # mask_layers: list of 1D boolean tensors (one per hidden layer)
        mask_layers = [masks_list[layer][sample_idx] for layer in range(len(masks_list))]
        A_w, b_w = _effective_affine_from_mask(model, mask_layers)
        # A_w: shape (2,), b_w: shape () or scalar
        affine_vecs[uid, :2] = A_w.numpy()
        affine_vecs[uid, 2] = b_w.item()

print(f"Affine vectors shape: {affine_vecs.shape}")

# ── Sanity check: K=N_patterns should reproduce float accuracy exactly ────────
# Map each test sample to its unique pattern uid
sample_uid = np.array([key_to_uid[k] for k in code_keys])

# Compute outputs using per-pattern affine maps
X_test_np = X_test.numpy()
y_approx_full = (
    affine_vecs[sample_uid, :2] * X_test_np
).sum(axis=1) + affine_vecs[sample_uid, 2]

y_test_np = y_test.numpy()
# Reference: float network sign
float_preds = (logits_test.squeeze(1).numpy() > 0).astype(float)
float_acc_frac = (float_preds == y_test_np).mean()
affine_full_acc = ((y_approx_full > 0).astype(float) == y_test_np).mean()

print(f"\nSanity check (K=N_patterns):")
print(f"  Float model accuracy:        {float_acc_frac:.4f}")
print(f"  Per-pattern affine accuracy: {affine_full_acc:.4f}")
if abs(float_acc_frac - affine_full_acc) > 0.01:
    print("  WARNING: sanity check failed — affine map mismatch > 1%")
else:
    print("  PASS: exact-affine reproduces float outputs.")

# ── K-means sweep ─────────────────────────────────────────────────────────────
K_candidates = [2, 4, 8, 12, 16, 24, 32, 48, 64]
K_candidates = [k for k in K_candidates if k <= N_patterns]
if N_patterns not in K_candidates:
    K_candidates.append(N_patterns)
K_candidates = sorted(set(K_candidates))

threshold = float_acc_frac - 0.01  # within 1 pp of float

per_K_accuracy = {}
K_star = None

print(f"\nK-means sweep (threshold = {threshold:.4f}):")
for K in K_candidates:
    if K == N_patterns:
        acc = affine_full_acc
    else:
        km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
        km.fit(affine_vecs)
        # Each unique pattern -> cluster label
        pattern_cluster = km.labels_  # (N_patterns,)
        # Centroid affine vectors
        centroids = km.cluster_centers_  # (K, 3)
        # For each test point: look up which cluster its uid belongs to
        test_cluster = pattern_cluster[sample_uid]  # (n_test,)
        # Reconstruct output using centroid affine map
        A_cent = centroids[test_cluster, :2]  # (n_test, 2)
        b_cent = centroids[test_cluster, 2]    # (n_test,)
        y_approx = (A_cent * X_test_np).sum(axis=1) + b_cent
        acc = ((y_approx > 0).astype(float) == y_test_np).mean()

    per_K_accuracy[K] = float(acc)
    marker = ""
    if K_star is None and acc >= threshold:
        K_star = K
        marker = " ← K*"
    print(f"  K={K:4d}  acc={acc:.4f}{marker}")

if K_star is None:
    K_star = N_patterns
    print(f"  Accuracy never recovered within 1% → K*=N_patterns (Inconclusive)")

ratio = K_star / N_patterns

# ── Verdict ───────────────────────────────────────────────────────────────────
if ratio <= 0.2:
    verdict = "CONFIRMED"
elif ratio > 0.7:
    verdict = "REFUTED"
else:
    verdict = "INCONCLUSIVE"

print(f"\n{'='*60}")
print(f"N_patterns = {N_patterns}")
print(f"K*         = {K_star}")
print(f"ratio      = {ratio:.4f}  ({ratio*100:.1f}%)")
print(f"float_acc  = {float_acc_frac:.4f}")
print(f"Verdict    = {verdict}")
print(f"{'='*60}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
results = {
    "N_patterns": int(N_patterns),
    "K_star": int(K_star),
    "ratio": float(ratio),
    "float_acc": float(float_acc_frac),
    "affine_full_acc": float(affine_full_acc),
    "per_K_accuracy": {str(k): float(v) for k, v in per_K_accuracy.items()},
    "arch": "2->32->32->1",
    "verdict": verdict,
    "seed": SEED,
    "n_train": int(n_train),
    "n_test": int(len(X_test)),
}
out_json = "/Users/mihai/Projects/quant_aware_training/autoresearch/results/cycle_012_kmeans_toy.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_json}")

# ── Save plot ─────────────────────────────────────────────────────────────────
Ks = sorted(per_K_accuracy.keys())
accs = [per_K_accuracy[k] for k in Ks]

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(Ks, accs, "o-", color="steelblue", lw=2, ms=6, label="k-means accuracy")
ax.axhline(float_acc_frac, color="green", ls="--", lw=1.5, label=f"Float acc ({float_acc_frac:.3f})")
ax.axhline(threshold, color="red", ls=":", lw=1.5, label=f"Float − 1% ({threshold:.3f})")
if K_star is not None:
    ax.axvline(K_star, color="orange", ls="-.", lw=1.5, label=f"K*={K_star} (ratio={ratio:.2f})")
ax.set_xlabel("K (number of clusters, log scale)")
ax.set_ylabel("Test accuracy")
ax.set_title(f"Jacobian k-means compression curve\n"
             f"N_patterns={N_patterns}, K*={K_star}, ratio={ratio:.2f}, verdict={verdict}")
ax.legend(loc="lower right")
ax.set_xlim(left=1)
ax.grid(True, alpha=0.3)
out_png = "/Users/mihai/Projects/quant_aware_training/autoresearch/results/cycle_012_kmeans_curve.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_png}")

print("\nDone.")
