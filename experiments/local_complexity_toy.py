"""
Cycle 023: Local complexity as compression difficulty predictor — toy ReLU

For each probe input x, count distinct activation patterns in an ε-ball in 2D input space.
Correlate per-layer local complexity with single-affine residual r from cycle 010.
Compare across training checkpoints (early=100, mid=500, late=2000).
"""
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

torch.manual_seed(42)
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
N_SAMPLES = 2000
N_TURNS = 3
SCALE = 2.0
HIDDEN = 64
N_LAYERS = 4
LR = 0.001
EPOCHS = 2000
CHECKPOINT_EPOCHS = [100, 500, 2000]
N_PROBE = 1000
EPS = 0.39  # ε for ε-ball in 2D input space
DEVICE = "cpu"

# Cycle 010 per-layer r_held values (layers 1–4)
R_HELD = [0.42967966198921204, 0.05038551986217499, 0.03330715000629425, 0.02373376302421093]

# ── Dataset ─────────────────────────────────────────────────────────────────
def make_spirals(n, n_turns, scale):
    n_half = n // 2
    t = torch.linspace(0, 1, n_half)
    angle = t * n_turns * 2 * torch.pi
    r = t * scale
    x0 = torch.stack([r * torch.cos(angle), r * torch.sin(angle)], dim=1)
    x1 = torch.stack([r * torch.cos(angle + torch.pi), r * torch.sin(angle + torch.pi)], dim=1)
    X = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(n_half), torch.ones(n_half)])
    perm = torch.randperm(n)
    return X[perm], y[perm]

X_all, y_all = make_spirals(N_SAMPLES, N_TURNS, SCALE)

# Probe set: first N_PROBE points from training data
X_probe = X_all[:N_PROBE]  # shape (1000, 2)

# ── Model ────────────────────────────────────────────────────────────────────
class ReLUMLP(torch.nn.Module):
    def __init__(self, hidden=64, n_layers=4):
        super().__init__()
        layers = [torch.nn.Linear(2, hidden)]
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(hidden, hidden))
        self.hidden_layers = torch.nn.ModuleList(layers)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.head(x).squeeze(-1)

    def get_activation_patterns(self, x):
        """Return list of binary patterns (one per hidden layer)."""
        patterns = []
        h = x
        for layer in self.hidden_layers:
            pre = layer(h)
            h = torch.relu(pre)
            pat = (pre > 0).int()  # shape (N, hidden)
            patterns.append(pat)
        return patterns  # list of length n_layers

# ── Training ─────────────────────────────────────────────────────────────────
model = ReLUMLP(HIDDEN, N_LAYERS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.BCEWithLogitsLoss()

checkpoints = {}  # epoch -> state_dict copy

print("Training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    logits = model(X_all)
    loss = loss_fn(logits, y_all)
    loss.backward()
    optimizer.step()

    if epoch in CHECKPOINT_EPOCHS:
        checkpoints[epoch] = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            acc = ((logits > 0) == y_all.bool()).float().mean().item()
        print(f"  Epoch {epoch:4d}: loss={loss.item():.4f}, acc={acc:.4f}")

print("Training complete.")

# ── Local Complexity ──────────────────────────────────────────────────────────
# Precompute pairwise distances in 2D input space (1000×1000)
print("Computing pairwise distances...")
X_np = X_probe.numpy()  # (N_PROBE, 2)
diff = X_np[:, None, :] - X_np[None, :, :]  # (N, N, 2)
dist2 = (diff ** 2).sum(axis=2)             # (N, N)
neighbor_mask = dist2 <= EPS ** 2           # boolean (N, N)
neighbor_counts = neighbor_mask.sum(axis=1)  # how many neighbors each point has (incl. self)
print(f"  Average neighbors per point: {neighbor_counts.mean():.1f} (target ~30)")

def compute_local_complexity(state_dict, layer_idx=None):
    """
    Returns per-layer local complexity for all hidden layers (or just one if layer_idx given).
    """
    model_tmp = ReLUMLP(HIDDEN, N_LAYERS)
    model_tmp.load_state_dict(state_dict)
    model_tmp.eval()

    with torch.no_grad():
        patterns_list = model_tmp.get_activation_patterns(X_probe)
        # patterns_list[l]: (N_PROBE, HIDDEN) int tensor

    lc_per_layer = []
    n_layers_to_check = range(N_LAYERS) if layer_idx is None else [layer_idx]

    for l in range(N_LAYERS):
        # Convert each row to a hashable tuple for pattern counting
        pat = patterns_list[l].numpy()  # (N_PROBE, HIDDEN)
        # For each probe point, count distinct patterns among neighbors
        local_counts = []
        for i in range(N_PROBE):
            neigh_idx = np.where(neighbor_mask[i])[0]
            neigh_pats = pat[neigh_idx]  # (k, HIDDEN)
            # Convert to set of tuples
            n_distinct = len(set(map(tuple, neigh_pats.tolist())))
            local_counts.append(n_distinct)
        lc_per_layer.append(float(np.mean(local_counts)))

    return lc_per_layer  # length N_LAYERS

results = {}
print("\nComputing local complexity at checkpoints...")
for epoch in CHECKPOINT_EPOCHS:
    print(f"  Checkpoint epoch={epoch}...")
    lc = compute_local_complexity(checkpoints[epoch])
    results[epoch] = lc
    print(f"    LC per layer: {[f'{v:.2f}' for v in lc]}")

# ── Correlation with r values ─────────────────────────────────────────────────
# Use late checkpoint (epoch 2000) — trained network local complexity
lc_late = results[2000]
rho, pval = spearmanr(lc_late, R_HELD)
print(f"\nSpearman(local_complexity @ epoch2000, r_held): rho={rho:.3f}, p={pval:.4f}")

# ── Training dynamics ─────────────────────────────────────────────────────────
lc_early_mean = float(np.mean(results[100]))
lc_late_mean = float(np.mean(results[2000]))
lc_mid_mean = float(np.mean(results[500]))
drop_pct = (lc_early_mean - lc_late_mean) / lc_early_mean * 100
print(f"\nMean LC: early={lc_early_mean:.2f}, mid={lc_mid_mean:.2f}, late={lc_late_mean:.2f}")
print(f"Drop early→late: {drop_pct:.1f}%")

# ── Evaluate criteria ─────────────────────────────────────────────────────────
corr_confirmed = rho >= 0.5
dynamics_confirmed = drop_pct >= 30.0

if corr_confirmed and dynamics_confirmed:
    outcome = "CONFIRMED"
elif not corr_confirmed and drop_pct < 5.0:
    outcome = "REFUTED"
else:
    outcome = "INCONCLUSIVE"

print(f"\nOutcome: {outcome}")
print(f"  Correlation criterion (rho≥0.5): {'PASS' if corr_confirmed else 'FAIL'} (rho={rho:.3f})")
print(f"  Dynamics criterion (drop≥30%): {'PASS' if dynamics_confirmed else 'FAIL'} ({drop_pct:.1f}%)")

# ── Save JSON ──────────────────────────────────────────────────────────────────
output = {
    "experiment": "local_complexity_toy_relu",
    "cycle": 23,
    "item": 29,
    "config": {
        "n_samples": N_SAMPLES,
        "n_turns": N_TURNS,
        "scale": SCALE,
        "architecture": f"2->{HIDDEN}x{N_LAYERS}->1",
        "lr": LR,
        "n_probe": N_PROBE,
        "epsilon": EPS,
        "avg_neighbors": float(neighbor_counts.mean()),
    },
    "checkpoints": {
        str(ep): results[ep] for ep in CHECKPOINT_EPOCHS
    },
    "spearman_rho": float(rho),
    "spearman_pval": float(pval),
    "r_held_cycle010": R_HELD,
    "lc_late": lc_late,
    "drop_pct_early_to_late": drop_pct,
    "corr_confirmed": bool(corr_confirmed),
    "dynamics_confirmed": bool(dynamics_confirmed),
    "outcome": outcome,
}
with open("autoresearch/results/cycle_023_local_complexity.json", "w") as f:
    json.dump(output, f, indent=2)

# ── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Panel 1: LC per layer at each checkpoint
ax = axes[0]
layers = [1, 2, 3, 4]
colors = ["#e74c3c", "#f39c12", "#2ecc71"]
labels = ["Early (ep 100)", "Mid (ep 500)", "Late (ep 2000)"]
for i, ep in enumerate(CHECKPOINT_EPOCHS):
    ax.plot(layers, results[ep], marker="o", color=colors[i], label=labels[i])
ax.set_xlabel("Layer")
ax.set_ylabel("Local complexity (mean distinct patterns)")
ax.set_title("Local complexity vs layer")
ax.set_xticks(layers)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: LC (late) vs r (cycle 010) scatter
ax = axes[1]
ax.scatter(lc_late, R_HELD, c="#2c3e50", zorder=5)
for i, (lc_i, r_i) in enumerate(zip(lc_late, R_HELD)):
    ax.annotate(f"L{i+1}", (lc_i, r_i), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.set_xlabel("Local complexity (late checkpoint)")
ax.set_ylabel("Single-affine residual r (cycle 010)")
ax.set_title(f"LC vs compression difficulty\nSpearman ρ={rho:.3f}, p={pval:.3f}")
ax.grid(True, alpha=0.3)

plt.suptitle(f"Cycle 023 — Local complexity ({outcome})", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("autoresearch/results/cycle_023_local_complexity.png", dpi=120, bbox_inches="tight")
plt.close()

print("\nSaved: autoresearch/results/cycle_023_local_complexity.json")
print("Saved: autoresearch/results/cycle_023_local_complexity.png")
