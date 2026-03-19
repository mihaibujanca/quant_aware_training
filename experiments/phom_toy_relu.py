"""
Cycle 047: Persistent homology of activation manifold through depth.
Trains a 2->32x4->1 ReLU spirals network, collects L1-L4 activations,
runs ripser PH on full 32D activations, checks Betti monotonicity,
and correlates with single-affine residual r from cycle 010.
"""

import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ripser import ripser as run_ripser

# ── reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── r values from cycle 010 (item 20) ─────────────────────────────────────────
R_VALUES = {1: 0.430, 2: 0.050, 3: 0.033, 4: 0.024}

# ── data ──────────────────────────────────────────────────────────────────────
def make_spirals(n=2000, turns=3, scale=2.0, noise=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n_each = n // 2
    t = np.linspace(0, turns * 2 * math.pi, n_each)
    r = t / (turns * 2 * math.pi)
    x0 = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    x1 = np.stack([r * np.cos(t + math.pi), r * np.sin(t + math.pi)], 1)
    x = np.concatenate([x0, x1], 0) * scale
    x += rng.randn(*x.shape) * noise
    y = np.concatenate([np.zeros(n_each), np.ones(n_each)], 0)
    perm = rng.permutation(n)
    return torch.tensor(x[perm], dtype=torch.float32), torch.tensor(y[perm], dtype=torch.float32)

# ── model ──────────────────────────────────────────────────────────────────────
class SpiralNet(nn.Module):
    def __init__(self, hidden=32, depth=4):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        # register hooks to capture pre-classifier activations
        self._acts = {}
        for i in range(depth):
            # index of ReLU output in Sequential: position 2*(i+1)-1 = 2i+1
            relu_idx = 2 * i + 1
            self.net[relu_idx].register_forward_hook(self._make_hook(i + 1))

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            self._acts[layer_idx] = output.detach()
        return hook

    def forward(self, x):
        self._acts.clear()
        return self.net(x)

    def get_activations(self):
        return dict(self._acts)


def train(model, X_tr, y_tr, epochs=2000, lr=0.001, batch=512):
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch):
            idx = perm[i:i+batch]
            xb, yb = X_tr[idx], y_tr[idx]
            opt.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
    return model


def accuracy(model, X, y):
    with torch.no_grad():
        logits = model(X).squeeze(1)
        preds = (logits > 0).float()
        return (preds == y).float().mean().item()


# ── PH utilities ──────────────────────────────────────────────────────────────
def total_persistence(diagrams, noise_frac=0.01):
    """
    Sum of lifetimes (death - birth) for H0 and H1 features with
    lifetime > noise_threshold = noise_frac * max_filtration_radius.
    """
    total = 0.0
    all_lifetimes = []
    for dgm in diagrams:
        finite = dgm[np.isfinite(dgm[:, 1])]
        lifetimes = finite[:, 1] - finite[:, 0]
        all_lifetimes.append(lifetimes)
    # global max filtration = max death value across all diagrams
    all_deaths = []
    for dgm in diagrams:
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) > 0:
            all_deaths.append(finite[:, 1].max())
    max_radius = max(all_deaths) if all_deaths else 1.0
    threshold = noise_frac * max_radius
    for lifetimes in all_lifetimes:
        total += float(lifetimes[lifetimes > threshold].sum())
    return total, threshold, max_radius


def compute_ph_for_activations(acts_np, max_radius_pct=0.95):
    """Run ripser on full-dim activations, return diagrams + total persistence."""
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(acts_np, metric='euclidean').astype(np.float32)
    max_r = float(np.percentile(D, max_radius_pct * 100))
    result = run_ripser(D, maxdim=1, thresh=max_r, distance_matrix=True)
    diagrams = result['dgms']
    total, thresh, max_rad = total_persistence(diagrams)
    return diagrams, total, thresh, max_rad


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    # Data
    X_all, y_all = make_spirals(n=2000, seed=42)
    X_test, y_test = make_spirals(n=500, seed=99)

    # Train model
    print("Training spiral network 2->32x4->1...")
    model = SpiralNet(hidden=32, depth=4)
    train(model, X_all, y_all, epochs=2000, lr=0.001)
    acc = accuracy(model, X_test, y_test)
    print(f"Test accuracy: {acc:.3f}")
    assert acc >= 0.95, f"Accuracy {acc:.3f} < 0.95 — undertrained, abort."

    # Random-init baseline
    print("Setting up random-init baseline...")
    model_rand = SpiralNet(hidden=32, depth=4)  # untrained

    # Collect activations on 500 test points
    model.eval()
    model_rand.eval()
    with torch.no_grad():
        _ = model(X_test)
        acts_trained = model.get_activations()
        _ = model_rand(X_test)
        acts_rand = model_rand.get_activations()

    # Convert to numpy
    acts_trained_np = {k: v.numpy() for k, v in acts_trained.items()}
    acts_rand_np = {k: v.numpy() for k, v in acts_rand.items()}

    # Gaussian null: 500 points in 32D, scale-matched per layer
    acts_gauss_np = {}
    for layer_idx in range(1, 5):
        mean_norm = float(np.linalg.norm(acts_trained_np[layer_idx], axis=1).mean())
        g = np.random.randn(500, 32).astype(np.float32)
        g = g / np.linalg.norm(g, axis=1, keepdims=True) * mean_norm
        acts_gauss_np[layer_idx] = g

    # Run PH
    results = {}
    print("Running PH on trained network activations...")
    trained_diagrams = {}
    trained_totals = {}
    for layer_idx in range(1, 5):
        print(f"  Layer {layer_idx}...")
        dgms, total, thresh, max_r = compute_ph_for_activations(acts_trained_np[layer_idx])
        trained_diagrams[layer_idx] = dgms
        trained_totals[layer_idx] = total
        print(f"    total persistence = {total:.4f}")

    print("Running PH on random-init activations...")
    rand_totals = {}
    for layer_idx in range(1, 5):
        print(f"  Layer {layer_idx}...")
        _, total, _, _ = compute_ph_for_activations(acts_rand_np[layer_idx])
        rand_totals[layer_idx] = total
        print(f"    total persistence = {total:.4f}")

    print("Running PH on Gaussian null activations...")
    gauss_totals = {}
    for layer_idx in range(1, 5):
        print(f"  Layer {layer_idx}...")
        _, total, _, _ = compute_ph_for_activations(acts_gauss_np[layer_idx])
        gauss_totals[layer_idx] = total
        print(f"    total persistence = {total:.4f}")

    # ── Monotonicity criterion ──────────────────────────────────────────────
    trained_vals = [trained_totals[i] for i in range(1, 5)]
    n_decreasing = sum(1 for i in range(len(trained_vals)-1)
                       if trained_vals[i+1] < trained_vals[i])
    n_transitions = len(trained_vals) - 1  # 3
    monotone_frac = n_decreasing / n_transitions
    monotone_criterion = monotone_frac >= 0.8  # >= 3/3 transitions or >= 2.4/3

    print(f"\nMonotonicity: {n_decreasing}/{n_transitions} transitions decrease "
          f"(frac={monotone_frac:.2f}, criterion={'MET' if monotone_criterion else 'NOT MET'})")

    # ── Correlation with r: rank-1 prediction ──────────────────────────────
    # Betti drop at layer i = trained_totals[i] - trained_totals[i+1]
    # (drop going from layer i to layer i+1, attributed to layer i+1)
    betti_drop = {}
    for i in range(1, 4):
        betti_drop[i+1] = trained_totals[i] - trained_totals[i+1]
    # Layer with largest Betti drop
    largest_drop_layer = max(betti_drop, key=betti_drop.get)
    # Layer with lowest r (L1=0.430, L2=0.050, L3=0.033, L4=0.024 -> lowest is L4)
    lowest_r_layer = min(R_VALUES, key=R_VALUES.get)
    rank1_correct = (largest_drop_layer == lowest_r_layer)

    print(f"\nLargest Betti drop: Layer {largest_drop_layer} "
          f"(drop={betti_drop[largest_drop_layer]:.4f})")
    print(f"Lowest r: Layer {lowest_r_layer} (r={R_VALUES[lowest_r_layer]:.3f})")
    print(f"Rank-1 prediction correct: {rank1_correct}")

    # Spearman on drops (L2, L3, L4) vs r (L2, L3, L4)
    from scipy.stats import spearmanr
    # Drop at L2,L3,L4; r at L2,L3,L4
    drop_vals = [betti_drop[2], betti_drop[3], betti_drop[4]]
    r_vals = [R_VALUES[2], R_VALUES[3], R_VALUES[4]]
    # Expectation: larger drop → lower r (easier compression) → negative correlation
    rho, pval = spearmanr(drop_vals, r_vals)
    print(f"\nSpearman(betti_drop, r) = {rho:.3f} (p={pval:.3f}, n=3)")

    # ── Null baseline validation ─────────────────────────────────────────────
    # Trained vs Gaussian at L1 and L4
    ratio_l1_gauss = trained_totals[1] / gauss_totals[1] if gauss_totals[1] > 0 else float('inf')
    ratio_l1_rand = trained_totals[1] / rand_totals[1] if rand_totals[1] > 0 else float('inf')
    ratio_l4_gauss = trained_totals[4] / gauss_totals[4] if gauss_totals[4] > 0 else float('inf')
    ratio_l4_rand = trained_totals[4] / rand_totals[4] if rand_totals[4] > 0 else float('inf')

    print(f"\nNull validation:")
    print(f"  Trained/Gaussian L1: {ratio_l1_gauss:.2f}x")
    print(f"  Trained/Random L1:   {ratio_l1_rand:.2f}x")
    print(f"  Trained/Gaussian L4: {ratio_l4_gauss:.2f}x")
    print(f"  Trained/Random L4:   {ratio_l4_rand:.2f}x")

    # ── Determine outcome ─────────────────────────────────────────────────────
    # Confirmed: monotone_criterion AND rank-1 prediction correct
    # Refuted: Betti complexity does NOT decrease or trained ≈ random
    # Inconclusive: simplifies overall but rank-1 wrong
    decreases_at_all = trained_vals[-1] < trained_vals[0]
    if monotone_criterion and rank1_correct:
        outcome = "confirmed"
    elif not decreases_at_all:
        outcome = "refuted"
    elif decreases_at_all and not monotone_criterion:
        outcome = "refuted"
    else:
        outcome = "inconclusive"

    print(f"\n=== OUTCOME: {outcome.upper()} ===")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        "test_accuracy": float(acc),
        "trained_totals": {str(k): float(v) for k, v in trained_totals.items()},
        "rand_totals": {str(k): float(v) for k, v in rand_totals.items()},
        "gauss_totals": {str(k): float(v) for k, v in gauss_totals.items()},
        "betti_drop": {str(k): float(v) for k, v in betti_drop.items()},
        "n_decreasing": int(n_decreasing),
        "n_transitions": int(n_transitions),
        "monotone_frac": float(monotone_frac),
        "monotone_criterion_met": bool(monotone_criterion),
        "largest_drop_layer": int(largest_drop_layer),
        "lowest_r_layer": int(lowest_r_layer),
        "rank1_correct": bool(rank1_correct),
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "ratio_l1_gauss": float(ratio_l1_gauss),
        "ratio_l1_rand": float(ratio_l1_rand),
        "ratio_l4_gauss": float(ratio_l4_gauss),
        "ratio_l4_rand": float(ratio_l4_rand),
        "r_values": R_VALUES,
        "outcome": outcome,
    }
    json_path = "autoresearch/results/cycle_047_phom_toy.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # ── 3-panel figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))

    # Panel A: persistence diagrams (2x2 grid for 4 layers) within a subplot area
    ax_title_a = fig.add_subplot(1, 3, 1)
    ax_title_a.axis('off')
    inner = matplotlib.gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=ax_title_a.get_subplotspec(), hspace=0.4, wspace=0.4
    )
    colors_h0 = '#2196F3'
    colors_h1 = '#E91E63'
    for i, layer_idx in enumerate(range(1, 5)):
        ax = fig.add_subplot(inner[i // 2, i % 2])
        dgms = trained_diagrams[layer_idx]
        # H0
        if len(dgms) > 0 and len(dgms[0]) > 0:
            finite_h0 = dgms[0][np.isfinite(dgms[0][:, 1])]
            if len(finite_h0):
                ax.scatter(finite_h0[:, 0], finite_h0[:, 1], s=10, c=colors_h0,
                           alpha=0.7, label='H0')
        # H1
        if len(dgms) > 1 and len(dgms[1]) > 0:
            finite_h1 = dgms[1][np.isfinite(dgms[1][:, 1])]
            if len(finite_h1):
                ax.scatter(finite_h1[:, 0], finite_h1[:, 1], s=10, c=colors_h1,
                           alpha=0.7, label='H1')
        # diagonal
        if len(dgms) > 0:
            all_finite = [dgms[d][np.isfinite(dgms[d][:, 1])] for d in range(len(dgms)) if len(dgms[d]) > 0]
            if all_finite:
                all_pts = np.concatenate(all_finite)
                lim = max(all_pts.max() * 1.05, 0.1)
                ax.set_xlim(0, lim); ax.set_ylim(0, lim)
                ax.plot([0, lim], [0, lim], 'k--', lw=0.5)
        ax.set_title(f'L{layer_idx} (TP={trained_totals[layer_idx]:.2f})', fontsize=8)
        ax.set_xlabel('birth', fontsize=7); ax.set_ylabel('death', fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)
    ax_title_a.set_title('(A) Persistence Diagrams', fontsize=10, y=1.02)

    # Panel B: Total persistence vs depth
    ax_b = fig.add_subplot(1, 3, 2)
    layers = list(range(1, 5))
    ax_b.plot(layers, [trained_totals[l] for l in layers], 'o-', color='#1565C0',
              lw=2, ms=8, label='Trained')
    ax_b.plot(layers, [rand_totals[l] for l in layers], 's--', color='#F57F17',
              lw=1.5, ms=7, label='Random-init')
    ax_b.plot(layers, [gauss_totals[l] for l in layers], '^:', color='#388E3C',
              lw=1.5, ms=7, label='Gaussian null')
    ax_b.set_xlabel('Layer', fontsize=11)
    ax_b.set_ylabel('Total Persistence Lifetime', fontsize=11)
    ax_b.set_title('(B) Betti Complexity vs Depth', fontsize=10)
    ax_b.legend(fontsize=9)
    ax_b.set_xticks(layers)
    mono_str = f"Monotone: {n_decreasing}/{n_transitions}"
    ax_b.text(0.05, 0.95, mono_str, transform=ax_b.transAxes, fontsize=9,
              verticalalignment='top',
              color='green' if monotone_criterion else 'red')

    # Panel C: PCA 2D visualization
    ax_c = fig.add_subplot(1, 3, 3)
    layer_colors = ['#E53935', '#8E24AA', '#1E88E5', '#43A047']
    for i, layer_idx in enumerate(range(1, 5)):
        acts = acts_trained_np[layer_idx]
        if acts.shape[1] > 2:
            pca = PCA(n_components=2)
            acts_2d = pca.fit_transform(acts)
        else:
            acts_2d = acts
        y_np = y_test.numpy()
        # plot class 0 as circles, class 1 as triangles
        mask0 = y_np == 0
        mask1 = y_np == 1
        ax_c.scatter(acts_2d[mask0, 0], acts_2d[mask0, 1], s=4,
                     c=[layer_colors[i]], alpha=0.3, marker='o')
        ax_c.scatter(acts_2d[mask1, 0], acts_2d[mask1, 1], s=4,
                     c=[layer_colors[i]], alpha=0.3, marker='^',
                     label=f'L{layer_idx}')
    ax_c.set_title('(C) PCA-2D Activations (visual only)', fontsize=10)
    ax_c.set_xlabel('PC1', fontsize=10); ax_c.set_ylabel('PC2', fontsize=10)
    ax_c.legend(fontsize=8, markerscale=2)

    plt.suptitle(
        f'Persistent Homology Through Depth — Toy ReLU Spirals\n'
        f'Outcome: {outcome.upper()} | Monotone {n_decreasing}/{n_transitions} | '
        f'Rank-1: {"correct" if rank1_correct else "wrong"} | '
        f'Spearman ρ={rho:.2f}',
        fontsize=11
    )
    plt.tight_layout()
    png_path = "autoresearch/results/cycle_047_phom_toy.png"
    plt.savefig(png_path, dpi=120, bbox_inches='tight')
    print(f"Saved figure: {png_path}")
    plt.close()

    return outcome, out


if __name__ == "__main__":
    outcome, data = main()
    print(f"\nFinal outcome: {outcome}")
    print(f"Trained totals: {data['trained_totals']}")
    print(f"Betti drop: {data['betti_drop']}")
