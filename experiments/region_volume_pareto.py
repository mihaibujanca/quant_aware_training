"""
Cycle 017 — Region Volume Pareto: do large regions carry most task accuracy? — toy ReLU.

Train a 2→32→32→1 spirals MLP (same as cycle 012), enumerate activation patterns on test
set, sort by volume (fraction of test points), compute cumulative accuracy curve, Gini
coefficient, and coverage-by-top-20% threshold.
"""

import json
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

RESULTS_DIR = "/Users/mihai/Projects/quant_aware_training/autoresearch/results"

# ── Dataset ───────────────────────────────────────────────────────────────────
X_np, y_np = make_spirals(n_samples=2000, noise=0.0, n_turns=3, random_state=SEED)
X_np = X_np * 2.0  # scale=2.0
y_np = y_np.astype(np.float32)

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32)

n_train = 1000
perm = torch.randperm(2000, generator=torch.Generator().manual_seed(SEED))
train_idx, test_idx = perm[:n_train], perm[n_train:]
X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

# ── Architecture ──────────────────────────────────────────────────────────────
class SimpleMLP(nn.Module):
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
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# ── Training ──────────────────────────────────────────────────────────────────
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
    print("ERROR: test accuracy below 90% — model underfitting, aborting.")
    sys.exit(1)

# ── Extract activation patterns on test set ───────────────────────────────────
with torch.no_grad():
    logits_test, masks_list, _, _ = _forward_hidden(model, X_test)

# flat_masks: (1000, 64) bool numpy; code_keys: tuple of 1000 hex strings
flat_masks, code_keys = _pack_activation_codes(masks_list)

unique_keys = list(dict.fromkeys(code_keys))
N_patterns = len(unique_keys)
print(f"\nUnique activation patterns on test set: {N_patterns}")

key_to_uid = {k: i for i, k in enumerate(unique_keys)}
uid_array  = np.array([key_to_uid[k] for k in code_keys])  # shape (1000,)

# Volume = count / 1000
counts = np.bincount(uid_array, minlength=N_patterns)
volumes = counts / len(X_test)

# ── Build A_ω + b_ω for each unique pattern ───────────────────────────────────
key_to_sample_idx = {}
for i, k in enumerate(code_keys):
    if k not in key_to_sample_idx:
        key_to_sample_idx[k] = i

# Store as affine map dicts: uid -> (A_w: (2,), b_w: scalar)
affine_maps = {}
with torch.no_grad():
    for uid, key in enumerate(unique_keys):
        sample_idx = key_to_sample_idx[key]
        mask_layers = [masks_list[layer][sample_idx] for layer in range(len(masks_list))]
        A_w, b_w = _effective_affine_from_mask(model, mask_layers)
        affine_maps[uid] = (A_w.numpy(), b_w.numpy() if hasattr(b_w, "numpy") else float(b_w))

# ── Sort regions by volume (descending) ───────────────────────────────────────
sorted_uids = np.argsort(-volumes)  # descending by volume

# Float model's correct/incorrect labels on test set
with torch.no_grad():
    float_preds = (model(X_test).squeeze(1) > 0).float().numpy()
float_correct = (float_preds == y_test.numpy())  # bool (1000,)
n_float_correct = float_correct.sum()
print(f"Float model: {int(n_float_correct)}/{len(X_test)} correct ({test_acc*100:.1f}%)")

# ── Precompute flat binary masks for Hamming fallback ─────────────────────────
# flat_masks shape: (1000, 64); one per test point
# We also need one canonical mask per unique pattern (from representative sample)
uid_canonical_mask = np.zeros((N_patterns, flat_masks.shape[1]), dtype=np.float32)
for uid, key in enumerate(unique_keys):
    sample_idx = key_to_sample_idx[key]
    uid_canonical_mask[uid] = flat_masks[sample_idx].astype(np.float32)

# ── Cumulative accuracy / coverage curve ─────────────────────────────────────
X_test_np = X_test.numpy()

# We'll evaluate at 0, 1, 2, ..., N_patterns kept regions
# To keep it tractable for large N, sample at every region (N_patterns ≤ ~500 expected)
accs = []
coverages = []  # fraction of test points whose true region is in kept set
top20_coverage_correctly_classified = None

kept_set = set()

for step in range(N_patterns + 1):
    if step > 0:
        kept_set.add(int(sorted_uids[step - 1]))

    if step == 0:
        # No regions kept — all fall back to top region, but we report step=0 as 0
        accs.append(float("nan"))
        coverages.append(0.0)
        continue

    # For each test point, find which region to use
    # If in kept set: use own uid's A_ω
    # Else: find nearest kept region by Hamming distance on canonical masks
    kept_list = np.array(sorted(kept_set), dtype=np.int64)
    kept_masks = uid_canonical_mask[kept_list]  # (k, 64)

    predictions = np.zeros(len(X_test_np), dtype=np.float32)
    in_kept = np.zeros(len(X_test_np), dtype=bool)

    for j in range(len(X_test_np)):
        own_uid = int(uid_array[j])
        if own_uid in kept_set:
            use_uid = own_uid
            in_kept[j] = True
        else:
            # Hamming distance to all kept canonical masks
            pt_mask = flat_masks[j].astype(np.float32)
            dists = np.sum(np.abs(kept_masks - pt_mask), axis=1)
            best_k = int(np.argmin(dists))
            use_uid = int(kept_list[best_k])

        A_w, b_w = affine_maps[use_uid]
        logit_val = float(A_w @ X_test_np[j]) + float(b_w)
        predictions[j] = 1.0 if logit_val > 0 else 0.0

    acc = float((predictions == y_test.numpy()).mean())
    cov = float(in_kept.mean())
    accs.append(acc)
    coverages.append(cov)

    # At 20% threshold: what fraction of float-correct points are in the kept set?
    thresh_k = math.ceil(0.20 * N_patterns)
    if step == thresh_k and top20_coverage_correctly_classified is None:
        if n_float_correct > 0:
            top20_coverage_correctly_classified = float(
                in_kept[float_correct].sum() / n_float_correct
            )
        else:
            top20_coverage_correctly_classified = 0.0

accs = accs[1:]   # drop step=0 placeholder
coverages = coverages[1:]

# ── Gini coefficient ──────────────────────────────────────────────────────────
def gini(v):
    v = np.sort(v)
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    cumsum = np.cumsum(v)
    return float(1.0 - 2.0 * cumsum.sum() / (n * cumsum[-1]) + 1.0 / n)

gini_coeff = gini(volumes)
print(f"\nGini coefficient of volume distribution: {gini_coeff:.4f}")

# Fraction of regions kept at top 20%
thresh_k = math.ceil(0.20 * N_patterns)
frac_test_covered = coverages[thresh_k - 1] if thresh_k <= len(coverages) else float("nan")
acc_at_20pct      = accs[thresh_k - 1]       if thresh_k <= len(accs) else float("nan")

print(f"Top 20% regions ({thresh_k} of {N_patterns}):")
print(f"  test points in those regions: {frac_test_covered*100:.1f}%")
print(f"  accuracy using piecewise-affine (with fallback): {acc_at_20pct*100:.1f}%")
print(f"  fraction of float-correct points in kept regions: {top20_coverage_correctly_classified*100:.1f}%")

# ── Verdict ───────────────────────────────────────────────────────────────────
if gini_coeff < 0.3:
    verdict = "REFUTED"
    reason  = f"Gini={gini_coeff:.3f} < 0.3 → volume distribution is approximately uniform"
elif top20_coverage_correctly_classified is not None and top20_coverage_correctly_classified >= 0.70:
    verdict = "CONFIRMED"
    reason  = (f"Top 20% regions cover {top20_coverage_correctly_classified*100:.1f}% "
               f"of float-correct points (threshold: 70%); Gini={gini_coeff:.3f}")
else:
    verdict = "INCONCLUSIVE"
    reason  = (f"Gini={gini_coeff:.3f} ≥ 0.3 (Pareto structure present) but top-20% "
               f"coverage of float-correct points = {top20_coverage_correctly_classified*100:.1f}% < 70%")

print(f"\nVerdict: {verdict}")
print(f"Reason:  {reason}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fracs_kept = np.arange(1, N_patterns + 1) / N_patterns

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: Volume distribution (sorted descending, log y)
ax = axes[0]
ax.bar(np.arange(N_patterns), volumes[sorted_uids], color="steelblue", width=1.0)
ax.set_yscale("log")
ax.set_xlabel("Region rank (by size)")
ax.set_ylabel("Volume (fraction of test set)")
ax.set_title(f"Region Volume Distribution\n(N={N_patterns}, Gini={gini_coeff:.3f})")
ax.axvline(thresh_k, color="red", linestyle="--", label=f"Top 20% ({thresh_k})")
ax.legend(fontsize=8)

# Panel 2: Cumulative accuracy curve
ax = axes[1]
ax.plot(fracs_kept * 100, np.array(accs) * 100, color="steelblue", lw=2, label="Accuracy")
ax.plot(fracs_kept * 100, np.array(coverages) * 100, color="orange", lw=1.5,
        linestyle="--", label="Coverage (in kept regions)")
ax.axvline(20, color="red", linestyle="--", alpha=0.7, label="20% threshold")
ax.axhline(test_acc * 100, color="gray", linestyle=":", alpha=0.7, label=f"Float acc ({test_acc*100:.1f}%)")
ax.set_xlabel("% regions kept (sorted by volume)")
ax.set_ylabel("% test points")
ax.set_title("Cumulative Accuracy + Coverage")
ax.legend(fontsize=8)
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)

# Panel 3: Lorenz curve
ax = axes[2]
sorted_vols = np.sort(volumes)
cumvol = np.concatenate([[0], np.cumsum(sorted_vols)])
cum_pop = np.linspace(0, 1, N_patterns + 1)
ax.plot(cum_pop * 100, cumvol * 100, color="steelblue", lw=2, label=f"Lorenz (Gini={gini_coeff:.3f})")
ax.plot([0, 100], [0, 100], color="gray", linestyle="--", lw=1, label="Perfect equality")
ax.fill_between(cum_pop * 100, cum_pop * 100, cumvol * 100, alpha=0.2, color="steelblue")
ax.set_xlabel("Cumulative % of regions (smallest first)")
ax.set_ylabel("Cumulative % of test-set volume")
ax.set_title("Lorenz Curve (Volume Inequality)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle(f"Cycle 017 — Region Volume Pareto | Verdict: {verdict}", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/cycle_017_pareto.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {RESULTS_DIR}/cycle_017_pareto.png")

# ── JSON results ─────────────────────────────────────────────────────────────
results = {
    "cycle": 17,
    "item": 34,
    "train_acc": round(train_acc, 4),
    "test_acc_float": round(test_acc, 4),
    "N_patterns": int(N_patterns),
    "N_test": int(len(X_test)),
    "gini_coeff": round(gini_coeff, 4),
    "thresh_k": int(thresh_k),
    "frac_test_covered_at_20pct": round(frac_test_covered, 4),
    "acc_at_20pct": round(acc_at_20pct, 4),
    "top20_coverage_float_correct": round(top20_coverage_correctly_classified, 4)
        if top20_coverage_correctly_classified is not None else None,
    "acc_at_all_regions": round(accs[-1], 4),
    "acc_curve": [round(a, 4) for a in accs],
    "coverage_curve": [round(c, 4) for c in coverages],
    "volume_sorted_desc": [round(float(v), 6) for v in volumes[sorted_uids]],
    "verdict": verdict,
    "reason": reason,
}

json_path = f"{RESULTS_DIR}/cycle_017_pareto.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {json_path}")
