"""
Cycle 016 — GELU PWL Approximation Fidelity
Item 32: toy GELU MLP on spirals, swap GELU → K-piece PWL, measure accuracy drop.
"""
import sys
sys.path.insert(0, "/Users/mihai/Projects/quant_aware_training")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aleph.datasets import make_spirals

torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cpu"

# ── Data ────────────────────────────────────────────────────────────────────
X_np, y_np = make_spirals(n_samples=2000, noise=0.5, n_turns=3, random_state=42)
X_np = X_np * 2.0
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

# 80/20 split
n_train = int(0.8 * len(X))
idx = torch.randperm(len(X))
X_tr, y_tr = X[idx[:n_train]], y[idx[:n_train]]
X_te, y_te = X[idx[n_train:]], y[idx[n_train:]]

# ── Fixed 100D projection ────────────────────────────────────────────────────
E = torch.randn(2, 100) / np.sqrt(100)  # (2, 100), fixed, not trained
E_buf = E  # keep in outer scope

def embed(x):
    return x @ E_buf  # (N, 100)

# ── GELU MLP ─────────────────────────────────────────────────────────────────
class GELUMLP(nn.Module):
    def __init__(self, in_dim=100, hidden=64, depth=4, use_embed=True):
        super().__init__()
        self.use_embed = use_embed
        layers = []
        dim = in_dim if use_embed else 2
        layers.append(nn.Linear(dim, hidden))
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
        self.hidden_layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden, 1)
        self.act = nn.GELU()

    def forward(self, x):
        if self.use_embed:
            x = embed(x)
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        return self.out(x)


def train(model, X_tr, y_tr, lr=0.001, epochs=5000, patience_acc=0.92):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_tr)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr)
        loss.backward()
        opt.step()
        if ep % 200 == 0:
            model.eval()
            with torch.no_grad():
                acc = ((torch.sigmoid(model(X_tr)) > 0.5).float() == y_tr).float().mean().item()
            if acc >= patience_acc:
                print(f"  ep {ep}: loss={loss.item():.4f}, train_acc={acc:.4f} — early stop")
                break
            if ep % 1000 == 0:
                print(f"  ep {ep}: loss={loss.item():.4f}, train_acc={acc:.4f}")
    return model


def eval_acc(model, X, y):
    model.eval()
    with torch.no_grad():
        return ((torch.sigmoid(model(X)) > 0.5).float() == y).float().mean().item()


# ── PWL GELU ──────────────────────────────────────────────────────────────────
def pwl_gelu(x: torch.Tensor, n_pieces: int) -> torch.Tensor:
    """
    K-piece piecewise-linear approximation of GELU.
    Domain: [-3, 3] with K+1 uniformly spaced breakpoints.
    Outside: g̃(x)=0 for x<-3, g̃(x)=x for x>3  (GELU tail behaviour).
    """
    bp = torch.linspace(-3.0, 3.0, n_pieces + 1, device=x.device)  # (K+1,)
    vals = F.gelu(bp)                                                 # (K+1,)

    # Clamp x into [-3, 3] for interpolation, then handle tails
    x_clamped = x.clamp(-3.0, 3.0)

    # Find which segment each value falls into: segment i spans [bp[i], bp[i+1]]
    # x_clamped is in [-3, 3], so segment index = floor((x_clamped - (-3)) / step)
    step = bp[1] - bp[0]
    seg = ((x_clamped - bp[0]) / step).long().clamp(0, n_pieces - 1)  # (N,)

    x0 = bp[seg]
    x1 = bp[seg + 1]
    v0 = vals[seg]
    v1 = vals[seg + 1]

    t = (x_clamped - x0) / (x1 - x0)  # linear weight within segment
    interp = v0 + t * (v1 - v0)

    # Tails
    result = torch.where(x < -3.0, torch.zeros_like(x),
             torch.where(x > 3.0, x, interp))
    return result


class PWLActivation(nn.Module):
    def __init__(self, n_pieces):
        super().__init__()
        self.n_pieces = n_pieces

    def forward(self, x):
        return pwl_gelu(x, self.n_pieces)


def swap_activation(model: GELUMLP, n_pieces: int) -> GELUMLP:
    """Return a copy of model with GELU swapped to n_pieces-PWL (no retraining)."""
    import copy
    m = copy.deepcopy(model)
    m.act = PWLActivation(n_pieces)
    return m


# ── Activation histogram check ────────────────────────────────────────────────
def check_activation_histogram(model, X):
    """Check fraction of activations outside [-3, 3]."""
    acts = []
    hooks = []
    def hook_fn(mod, inp, out):
        acts.append(inp[0].detach().flatten())
    for layer in model.hidden_layers:
        hooks.append(layer.register_forward_hook(hook_fn))
    model.eval()
    with torch.no_grad():
        model(X)
    for h in hooks:
        h.remove()
    all_acts = torch.cat(acts)
    frac_outside = ((all_acts < -3) | (all_acts > 3)).float().mean().item()
    return frac_outside, all_acts


# ── Main ──────────────────────────────────────────────────────────────────────
print("=== Training GELU MLP (100D embedding) ===")
model_100d = GELUMLP(in_dim=100, hidden=64, depth=4, use_embed=True)
model_100d = train(model_100d, X_tr, y_tr)
acc_float_100d = eval_acc(model_100d, X_te, y_te)
print(f"Float GELU (100D) test acc: {acc_float_100d:.4f}")

# Check if 100D model trained well (threshold 88%)
USE_100D = acc_float_100d >= 0.88
if not USE_100D:
    print(f"  100D acc={acc_float_100d:.3f} < 0.88 — falling back to 2D-only model")
    print("\n=== Training GELU MLP (2D input, no embedding) ===")
    model_2d = GELUMLP(in_dim=2, hidden=64, depth=4, use_embed=False)
    model_2d = train(model_2d, X_tr, y_tr)
    acc_float_2d = eval_acc(model_2d, X_te, y_te)
    print(f"Float GELU (2D) test acc: {acc_float_2d:.4f}")
    main_model = model_2d
    float_acc = acc_float_2d
    model_label = "2D"
else:
    main_model = model_100d
    float_acc = acc_float_100d
    model_label = "100D"

# Activation histogram
frac_outside, all_acts = check_activation_histogram(main_model, X_te)
print(f"\nActivation histogram: {frac_outside*100:.2f}% inputs outside [-3, 3] domain")

# ── PWL sweep ────────────────────────────────────────────────────────────────
K_sweep = [1, 3, 5, 10]
results = {}
print(f"\n=== PWL Sweep (model: {model_label}, float acc={float_acc:.4f}) ===")
for K in K_sweep:
    m_pwl = swap_activation(main_model, K)
    acc_pwl = eval_acc(m_pwl, X_te, y_te)
    drop_abs = float_acc - acc_pwl
    drop_rel = drop_abs / float_acc * 100
    results[K] = {"acc": acc_pwl, "drop_abs": drop_abs, "drop_pct": drop_rel}
    print(f"  K={K:2d}: acc={acc_pwl:.4f}  drop={drop_abs*100:.2f}pp  ({drop_rel:.2f}% rel)")

# Also report for 100D model if we used 2D fallback
if not USE_100D:
    print(f"\n  [Also reporting 100D model for reference: float acc={acc_float_100d:.4f}]")
    for K in K_sweep:
        m_pwl = swap_activation(model_100d, K)
        acc_pwl = eval_acc(m_pwl, X_te, y_te)
        drop_abs = acc_float_100d - acc_pwl
        print(f"    100D K={K:2d}: acc={acc_pwl:.4f}  drop={drop_abs*100:.2f}pp")

# ── Outcome ───────────────────────────────────────────────────────────────────
drop3 = results[3]["drop_pct"]
drop5 = results[5]["drop_pct"]
drop10 = results[10]["drop_pct"]

if drop3 < 3.0 and drop5 < 1.0:
    outcome = "CONFIRMED"
    detail = f"3-piece drop={drop3:.2f}% < 3%, 5-piece drop={drop5:.2f}% < 1%"
elif drop10 > 5.0:
    outcome = "REFUTED"
    detail = f"10-piece drop={drop10:.2f}% > 5%"
elif results[3]["drop_pct"] >= 3.0 or results[5]["drop_pct"] >= 1.0:
    outcome = "INCONCLUSIVE"
    detail = f"3-piece drop={drop3:.2f}%, 5-piece drop={drop5:.2f}%"
else:
    outcome = "INCONCLUSIVE"
    detail = f"drops: {[results[k]['drop_pct'] for k in K_sweep]}"

print(f"\n=== OUTCOME: {outcome} ===")
print(f"  {detail}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: accuracy vs K
ax = axes[0]
ks = K_sweep
accs = [results[k]["acc"] for k in ks]
drops = [results[k]["drop_pct"] for k in ks]
ax.plot([0] + ks, [float_acc] + accs, "o-", color="steelblue", linewidth=2, markersize=7, label="Test accuracy")
ax.axhline(float_acc, color="gray", linestyle="--", alpha=0.6, label="Float GELU")
ax.axhline(float_acc - 0.03, color="orange", linestyle=":", alpha=0.8, label="3% drop threshold")
ax.axhline(float_acc - 0.01, color="green", linestyle=":", alpha=0.8, label="1% drop threshold")
ax.set_xlabel("Number of PWL pieces (K)")
ax.set_ylabel("Test accuracy")
ax.set_title(f"GELU → K-piece PWL (model: {model_label})\nFloat acc = {float_acc:.4f}")
ax.set_xticks([0] + ks)
ax.set_xticklabels(["float\n(GELU)"] + [str(k) for k in ks])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: drop in pp
ax2 = axes[1]
drops_abs = [results[k]["drop_abs"] * 100 for k in ks]
bars = ax2.bar([str(k) for k in ks], drops_abs, color=["#e74c3c" if d > 3 else "#f39c12" if d > 1 else "#2ecc71" for d in drops])
ax2.axhline(3.0, color="orange", linestyle="--", alpha=0.8, label="3pp threshold")
ax2.axhline(1.0, color="green", linestyle="--", alpha=0.8, label="1pp threshold")
ax2.set_xlabel("Number of PWL pieces (K)")
ax2.set_ylabel("Accuracy drop (percentage points)")
ax2.set_title(f"Accuracy drop vs K\n({frac_outside*100:.2f}% activations outside [-3,3])")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")
for bar, d in zip(bars, drops_abs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{d:.2f}pp", ha="center", va="bottom", fontsize=9)

outcome_color = {"CONFIRMED": "green", "REFUTED": "red", "INCONCLUSIVE": "orange"}[outcome]
fig.suptitle(f"Cycle 016 — GELU PWL Approximation: {outcome}", fontsize=13, color=outcome_color, fontweight="bold")
plt.tight_layout()
plot_path = "autoresearch/results/cycle_016_gelu_pwl_approximation.png"
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
print(f"\nPlot saved to {plot_path}")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n=== Summary Table ===")
print(f"{'K':>5}  {'acc':>8}  {'drop_pp':>8}  {'drop_%':>8}")
print(f"{'float':>5}  {float_acc:>8.4f}  {'—':>8}  {'—':>8}")
for K in K_sweep:
    r = results[K]
    print(f"{K:>5}  {r['acc']:>8.4f}  {r['drop_abs']*100:>8.2f}  {r['drop_pct']:>8.2f}")
