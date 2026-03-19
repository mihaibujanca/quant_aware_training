"""
Cycle 065: Topological Truncation — Betti-Guided Depth Reduction
Tests whether Betti numbers stabilize at layer T, allowing post-T affine replacement.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ripser
from sklearn.datasets import make_moons

DEVICE = "cpu"
EPS = 1e-8


# ── Data ──────────────────────────────────────────────────────────────────────

def make_spirals(n=2000, turns=3, scale=2.0, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n_half = n // 2
    t = torch.linspace(0, turns * 2 * np.pi, n_half)
    x0 = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1) * scale / (turns * 2 * np.pi)
    x1 = torch.stack([t * torch.cos(t + np.pi), t * torch.sin(t + np.pi)], dim=1) * scale / (turns * 2 * np.pi)
    X = torch.cat([x0, x1], dim=0).float()
    X += torch.tensor(rng.normal(0, noise, X.shape), dtype=torch.float32)
    y = torch.cat([torch.zeros(n_half), torch.ones(n_half)]).float()
    perm = torch.randperm(n)
    return X[perm], y[perm]


def make_moons_data(n=2000, noise=0.1, seed=42):
    rng_np = np.random.default_rng(seed)
    X_np, y_np = make_moons(n_samples=n, noise=noise, random_state=seed)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    return X, y


# ── Model ─────────────────────────────────────────────────────────────────────

class ReLUNet(nn.Module):
    def __init__(self, in_dim=2, hidden=32, depth=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.hidden_layers = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.hidden_layers(x)).squeeze(-1)


def get_layer_activations(model, X):
    """Return post-ReLU activations for each of the 4 hidden layers."""
    acts = []
    x = X
    for i, module in enumerate(model.hidden_layers):
        x = module(x)
        if isinstance(module, nn.ReLU):
            acts.append(x.detach().clone())
    return acts  # list of 4 tensors [N, 32]


def train(model, X, y, epochs=2000, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X)) > 0.5).float()
        acc = (preds == y).float().mean().item()
    return acc


# ── Ripser / Betti ────────────────────────────────────────────────────────────

def subsample_stratified(X_np, y_np, n=500, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.unique(y_np)
    idx = []
    per_class = n // len(classes)
    for c in classes:
        ci = np.where(y_np == c)[0]
        chosen = rng.choice(ci, size=min(per_class, len(ci)), replace=False)
        idx.append(chosen)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X_np[idx]


def total_persistence(dgm):
    """Sum of lifetimes for a persistence diagram (Nx2 array). Returns 0 for empty."""
    if len(dgm) == 0:
        return 0.0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return 0.0
    return float(np.sum(finite[:, 1] - finite[:, 0]))


def compute_betti_per_layer(acts_list, y_np):
    """For each layer's activations, run ripser and return TP_0, TP_1."""
    results = []
    for act in acts_list:
        A = act.numpy()
        A_sub = subsample_stratified(A, y_np, n=500, seed=0)
        # diameter = max pairwise distance (use max of per-row norms as upper bound)
        dists = np.sqrt(((A_sub[:, None] - A_sub[None, :]) ** 2).sum(-1))
        thresh = float(dists.max())
        dgms = ripser.ripser(A_sub, maxdim=1, thresh=thresh)["dgms"]
        tp0 = total_persistence(dgms[0])
        tp1 = total_persistence(dgms[1])
        results.append((tp0, tp1))
    return results  # list of (tp0, tp1) per layer


# ── Stabilization ─────────────────────────────────────────────────────────────

def find_stabilization(tp_per_layer, tol=0.10):
    """
    Find earliest T such that for all j >= T:
      |TP_k(j) - TP_k(T)| / TP_k(T) < tol for k in {0,1}.
    Division-by-zero: if TP_k(T) < EPS, that dim is trivially stable.
    Returns T (0-indexed) or None.
    """
    L = len(tp_per_layer)
    for T in range(L):
        tp0_T, tp1_T = tp_per_layer[T]
        stable = True
        for j in range(T, L):
            tp0_j, tp1_j = tp_per_layer[j]
            # H0
            if tp0_T >= EPS:
                rel0 = abs(tp0_j - tp0_T) / tp0_T
                if rel0 >= tol:
                    stable = False
                    break
            # H1
            if tp1_T >= EPS:
                rel1 = abs(tp1_j - tp1_T) / tp1_T
                if rel1 >= tol:
                    stable = False
                    break
        if stable:
            return T
    return None


# ── Affine Replacement ────────────────────────────────────────────────────────

def affine_accuracy(model, X, y, T_layer):
    """
    Fit affine map from post-ReLU activations at layer T to float output logits.
    Return accuracy of truncated network.
    T_layer: 0-indexed (0 = after layer 1, ..., 3 = after layer 4).
    """
    with torch.no_grad():
        acts = get_layer_activations(model, X)
        A_T = acts[T_layer].numpy()  # [N, 32]
        logits_float = model(X).numpy()  # [N]

    # Fit: logits ≈ A_T @ w + b  (add bias column)
    A_aug = np.hstack([A_T, np.ones((len(A_T), 1))])  # [N, 33]
    sol, _, _, _ = np.linalg.lstsq(A_aug, logits_float, rcond=None)
    w, b = sol[:-1], sol[-1]

    preds_aff = torch.sigmoid(torch.tensor(A_T @ w + b, dtype=torch.float32))
    acc_aff = ((preds_aff > 0.5).float() == y).float().mean().item()
    acc_float = ((torch.sigmoid(model(X)) > 0.5).float() == y).float().mean().item()
    return acc_aff, acc_float


# ── Main ──────────────────────────────────────────────────────────────────────

SEEDS = [42, 123, 7]
DATASETS = {"spirals": make_spirals, "moons": make_moons_data}
DEPTH = 4
HIDDEN = 32

all_results = {}

for ds_name, ds_fn in DATASETS.items():
    ds_results = []
    for seed in SEEDS:
        print(f"\n=== {ds_name} seed={seed} ===")
        X, y = ds_fn(n=2000, seed=seed) if ds_name == "spirals" else ds_fn(n=2000, noise=0.1, seed=seed)
        X, y = X.to(DEVICE), y.to(DEVICE)

        model = ReLUNet(in_dim=2, hidden=HIDDEN, depth=DEPTH).to(DEVICE)
        acc_train = train(model, X, y)
        print(f"  Train acc: {acc_train:.3f}")
        if acc_train < 0.90:
            print(f"  WARNING: acc {acc_train:.3f} < 0.90, skipping seed")
            ds_results.append({"seed": seed, "skipped": True, "acc_train": acc_train})
            continue

        y_np = y.numpy()
        acts = get_layer_activations(model, X)
        tp_per_layer = compute_betti_per_layer(acts, y_np)
        print(f"  TP per layer: {[(f'{tp0:.3f}', f'{tp1:.3f}') for tp0, tp1 in tp_per_layer]}")

        T = find_stabilization(tp_per_layer)
        print(f"  Stabilization T={T}")

        aff_results = {}
        for candidate_T in range(DEPTH):
            acc_aff, acc_float = affine_accuracy(model, X, y, candidate_T)
            acc_drop = acc_float - acc_aff
            aff_results[candidate_T] = {"acc_aff": acc_aff, "acc_float": acc_float, "drop": acc_drop}
            print(f"  T={candidate_T}: float={acc_float:.3f} aff={acc_aff:.3f} drop={acc_drop:.3f}")

        ds_results.append({
            "seed": seed,
            "skipped": False,
            "acc_train": acc_train,
            "tp_per_layer": tp_per_layer,
            "T_stabilization": T,
            "affine": aff_results,
        })

    all_results[ds_name] = ds_results

# ── Criterion Evaluation ──────────────────────────────────────────────────────

print("\n=== CRITERION EVALUATION ===")
confirmed_datasets = []
for ds_name, ds_results in all_results.items():
    valid = [r for r in ds_results if not r.get("skipped")]
    if len(valid) < 2:
        print(f"{ds_name}: insufficient valid seeds")
        continue
    # Check T <= L-2 = 2 (0-indexed: T in {0,1,2} for L=4)
    T_values = [r["T_stabilization"] for r in valid]
    T_ok = sum(1 for T in T_values if T is not None and T <= DEPTH - 2) >= 2
    # Check affine loss <= 5% at T
    aff_ok = 0
    for r in valid:
        T = r["T_stabilization"]
        if T is not None and T <= DEPTH - 2:
            drop = r["affine"][T]["drop"]
            if drop <= 0.05:
                aff_ok += 1
    aff_criterion = aff_ok >= 2
    print(f"{ds_name}: T_values={T_values} T_ok={T_ok} aff_ok={aff_ok}/3 => {'CONFIRMED' if T_ok and aff_criterion else 'NOT CONFIRMED'}")
    if T_ok and aff_criterion:
        confirmed_datasets.append(ds_name)

if len(confirmed_datasets) == 2:
    outcome = "CONFIRMED"
elif len(confirmed_datasets) == 0:
    # Check if refuted (never stabilizes or large affine loss)
    all_T_none = all(
        all(r.get("T_stabilization") is None for r in all_results[ds] if not r.get("skipped"))
        for ds in all_results
    )
    outcome = "REFUTED" if all_T_none else "INCONCLUSIVE"
else:
    outcome = "INCONCLUSIVE"
print(f"\nOUTCOME: {outcome}")

# ── Save JSON ─────────────────────────────────────────────────────────────────

# Make serializable
def make_serial(obj):
    if isinstance(obj, dict):
        return {str(k): make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serial(v) for v in obj]
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

results_out = {
    "outcome": outcome,
    "confirmed_datasets": confirmed_datasets,
    "data": make_serial(all_results),
}
with open("autoresearch/results/cycle_065_topo_truncation.json", "w") as f:
    json.dump(results_out, f, indent=2)
print("Saved JSON.")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Cycle 065: Total Persistence per Layer (Betti-guided depth reduction)", fontsize=12)

for ax, (ds_name, ds_results) in zip(axes, all_results.items()):
    valid = [r for r in ds_results if not r.get("skipped")]
    for r in valid:
        tp = r["tp_per_layer"]
        layers = list(range(1, len(tp) + 1))
        tp0 = [t[0] for t in tp]
        tp1 = [t[1] for t in tp]
        seed = r["seed"]
        T = r["T_stabilization"]
        ax.plot(layers, tp0, "o-", label=f"H0 s={seed} T={T}")
        ax.plot(layers, tp1, "s--", alpha=0.6, label=f"H1 s={seed}")
    ax.set_title(f"{ds_name}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Total Persistence")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("autoresearch/results/cycle_065_topo_truncation.png", dpi=120, bbox_inches="tight")
print("Saved plot.")
print("Done.")
