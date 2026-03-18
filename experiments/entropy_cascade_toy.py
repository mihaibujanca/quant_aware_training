"""
Cycle 019: Activation pattern entropy cascade — toy ReLU
Architecture: 2 → 32 → 32 → 32 → 32 → 1
"""

import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aleph.datasets import make_spirals

# ── config ─────────────────────────────────────────────────────────────────
N_TRAIN = 2000
N_TEST  = 500
N_TURNS = 3
SCALE   = 2.0
HIDDEN  = 32
N_LAYERS = 4
LR      = 1e-3
EPOCHS  = 3000
SEED    = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── dataset ─────────────────────────────────────────────────────────────────
X_train, y_train = make_spirals(N_TRAIN, n_turns=N_TURNS, noise=0.1)
X_test,  y_test  = make_spirals(N_TEST,  n_turns=N_TURNS, noise=0.1)
X_train = torch.tensor(X_train, dtype=torch.float32) * SCALE
X_test  = torch.tensor(X_test,  dtype=torch.float32) * SCALE
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

# ── model ────────────────────────────────────────────────────────────────────
class ReLUMLP(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, out_dim=1):
        super().__init__()
        layers = []
        prev = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = ReLUMLP(2, HIDDEN, N_LAYERS)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()

# ── training ─────────────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_train), y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    logits = model(X_test)
    acc = ((logits > 0).float() == y_test).float().mean().item()
print(f"Test accuracy: {acc:.4f}")
assert acc > 0.85, f"Model underfit: acc={acc:.4f}"

# ── extract pre-ReLU activations on test set ─────────────────────────────────
pre_relu = []  # list of (N, 32) tensors per layer
hooks = []

linear_layers = [m for m in model.net if isinstance(m, nn.Linear)][:-1]  # exclude output

def make_hook(storage):
    def hook(module, inp, out):
        storage.append(out.detach())
    return hook

for lyr in linear_layers:
    h = lyr.register_forward_hook(make_hook(pre_relu))
    hooks.append(h)

with torch.no_grad():
    _ = model(X_test)

for h in hooks:
    h.remove()

assert len(pre_relu) == N_LAYERS, f"Expected {N_LAYERS} layers, got {len(pre_relu)}"

# binary activation masks q_i = (z_i > 0)
masks = [(z > 0).numpy().astype(bool) for z in pre_relu]  # list of (N, 32) bool arrays

# ── entropy helper ────────────────────────────────────────────────────────────
def pattern_entropy_bits(binary_matrix):
    """Compute Shannon entropy in bits of row-pattern distribution."""
    N = binary_matrix.shape[0]
    # Convert rows to tuples for hashing
    rows = [tuple(r.tolist()) for r in binary_matrix]
    from collections import Counter
    counts = Counter(rows)
    total = N
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * math.log2(p)
    return H

# ── compute entropy cascade ──────────────────────────────────────────────────
# Marginal entropies
H_marginal = []
for i, m in enumerate(masks):
    h = pattern_entropy_bits(m)
    H_marginal.append(h)
    print(f"H(L{i+1}) = {h:.4f} bits  (distinct patterns: {len(set(tuple(r.tolist()) for r in m))})")

# Joint entropies via concatenation
joint_masks = [masks[0]]
H_joint = [H_marginal[0]]
for i in range(1, N_LAYERS):
    concat = np.concatenate(masks[:i+1], axis=1)
    joint_masks.append(concat)
    h_joint = pattern_entropy_bits(concat)
    H_joint.append(h_joint)
    n_distinct = len(set(tuple(r.tolist()) for r in concat))
    print(f"H(L1..L{i+1}) = {h_joint:.4f} bits  (distinct joint patterns: {n_distinct})")

# Conditional entropies
# cH(L1) = H(L1) (marginal, no conditioning)
# cH(Li | L1..L{i-1}) = H(L1..Li) - H(L1..L{i-1})
cH = [H_joint[0]]  # layer 1 has no conditioning
for i in range(1, N_LAYERS):
    ch = H_joint[i] - H_joint[i-1]
    cH.append(ch)

print("\n=== Entropy Cascade ===")
for i, (hm, ch) in enumerate(zip(H_marginal, cH)):
    print(f"Layer {i+1}: H(marginal)={hm:.4f}  cH(L{i+1}|prev)={ch:.4f}  ratio={ch/H_marginal[0]:.3f}")

# ── single-affine residuals (inline, same architecture) ──────────────────────
# r_held = unexplained variance fraction when fitting single affine map to each layer's I/O
# Get layer inputs by hooking right before each linear (= after previous relu, or raw input)
layer_inputs = [X_test.numpy()]  # layer 0 input = raw data

relu_outputs = []
relu_hooks = []
relu_layers = [m for m in model.net if isinstance(m, nn.ReLU)]

def make_relu_hook(storage):
    def hook(module, inp, out):
        storage.append(out.detach().numpy())
    return hook

for rl in relu_layers:
    rh = rl.register_forward_hook(make_relu_hook(relu_outputs))
    relu_hooks.append(rh)

with torch.no_grad():
    _ = model(X_test)

for rh in relu_hooks:
    rh.remove()

# layer i input = relu_outputs[i-1] for i>0, else raw X
layer_inputs = [X_test.numpy()] + relu_outputs[:-1]

r_vals = []
for i in range(N_LAYERS):
    X_in  = layer_inputs[i]           # (N, d_in)
    X_out = relu_outputs[i]           # (N, 32)
    # Fit affine: X_out ≈ X_in @ A^T + b via lstsq
    X_aug = np.hstack([X_in, np.ones((len(X_in), 1))])
    coef, _, _, _ = np.linalg.lstsq(X_aug, X_out, rcond=None)
    X_pred = X_aug @ coef
    ss_res = np.sum((X_out - X_pred)**2)
    ss_tot = np.sum((X_out - X_out.mean(axis=0))**2)
    r = ss_res / ss_tot if ss_tot > 0 else 0.0
    r_vals.append(r)
    print(f"Layer {i+1} single-affine residual r = {r:.4f}")

# ── correlation analysis ──────────────────────────────────────────────────────
from scipy.stats import spearmanr
drop_ratios = [ch / H_marginal[0] for ch in cH]
rho, pval = spearmanr(drop_ratios, r_vals)
print(f"\nSpearman(conditional_H_ratio, r_affine) = {rho:.4f}  p={pval:.4f}")

# ── confirmation logic ────────────────────────────────────────────────────────
min_drop = min(drop_ratios[1:])  # skip layer 1 (no conditioning)
min_drop_layer = drop_ratios[1:].index(min_drop) + 2  # 1-indexed
min_drop_r = r_vals[min_drop_layer - 1]
median_r = float(np.median(r_vals))

print(f"\nMin conditional entropy drop ratio: {min_drop:.3f} at layer {min_drop_layer}")
print(f"r_affine at that layer: {min_drop_r:.4f}  median r: {median_r:.4f}")

# Confirmation: min conditional entropy drop ≤ 0.5 AND low-cH layers have r much lower
# than layer 1 (i.e., r at low-cH layer < r_L1 * 0.5, to avoid median sensitivity with n=4)
r_L1 = r_vals[0]
low_cH_layers = [i for i, d in enumerate(drop_ratios) if i > 0 and d <= 0.5]
low_cH_and_low_r = [i for i in low_cH_layers if r_vals[i] < r_L1 * 0.5]

if min_drop <= 0.5 and len(low_cH_and_low_r) > 0:
    verdict = "confirmed"
    reason = (f"Layer {min_drop_layer} conditional H ratio={min_drop:.3f} ≤ 0.5 "
              f"AND {len(low_cH_and_low_r)} low-cH layer(s) have r < 0.5×r_L1 "
              f"(r_L1={r_L1:.4f}); low-r layers: {[i+1 for i in low_cH_and_low_r]}")
elif all(d > 0.8 for d in drop_ratios[1:]):
    verdict = "refuted"
    reason = "All conditional entropy drop ratios > 0.8 — every layer adds substantial new information"
else:
    verdict = "inconclusive"
    reason = f"Entropy varies (min ratio={min_drop:.3f}) but r correlation rho={rho:.4f}"

print(f"\nVerdict: {verdict.upper()}")
print(f"Reason: {reason}")

# ── save results ──────────────────────────────────────────────────────────────
results = {
    "experiment": "activation_pattern_entropy_cascade",
    "cycle": 19,
    "item": 37,
    "config": {
        "architecture": f"2->{HIDDEN}x{N_LAYERS}->1",
        "hidden": HIDDEN,
        "n_layers": N_LAYERS,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_turns": N_TURNS,
        "scale": SCALE,
        "lr": LR,
        "epochs": EPOCHS,
    },
    "test_accuracy": acc,
    "marginal_entropy_bits": H_marginal,
    "joint_entropy_bits": H_joint,
    "conditional_entropy_bits": cH,
    "conditional_drop_ratio": drop_ratios,
    "single_affine_residuals": r_vals,
    "spearman_rho": float(rho),
    "spearman_pval": float(pval),
    "min_drop_ratio": float(min_drop),
    "min_drop_layer": int(min_drop_layer),
    "min_drop_r": float(min_drop_r),
    "median_r": float(median_r),
    "verdict": verdict,
    "verdict_reason": reason,
}

with open("autoresearch/results/cycle_019_entropy_cascade.json", "w") as f:
    json.dump(results, f, indent=2)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(8, 5))
layers = [f"L{i+1}" for i in range(N_LAYERS)]
x = np.arange(N_LAYERS)
width = 0.35

bars1 = ax1.bar(x - width/2, H_marginal, width, label="H(Li) marginal", color="steelblue", alpha=0.8)
bars2 = ax1.bar(x + width/2, cH, width, label="cH(Li|L<i) conditional", color="tomato", alpha=0.8)
ax1.set_xlabel("Layer")
ax1.set_ylabel("Entropy (bits)")
ax1.set_xticks(x)
ax1.set_xticklabels(layers)
ax1.set_title(f"Activation Pattern Entropy Cascade — spirals 2→{HIDDEN}×{N_LAYERS}→1\n"
              f"Test acc={acc:.3f}  Verdict: {verdict}")

ax2 = ax1.twinx()
ax2.plot(x, r_vals, "k--o", label="Single-affine residual r", linewidth=1.5, markersize=6)
ax2.set_ylabel("Single-affine residual r (frac. unexplained var)")
ax2.set_ylim(0, max(r_vals) * 1.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("autoresearch/results/cycle_019_entropy_cascade.png", dpi=120)
print("Plot saved.")

# ── summary ────────────────────────────────────────────────────────────────────
summary = f"""# Cycle 019 Summary — Activation Pattern Entropy Cascade (Toy ReLU)

## Config
Architecture: 2→{HIDDEN}×{N_LAYERS}→1 ReLU MLP
Dataset: spirals n={N_TRAIN} test={N_TEST} turns={N_TURNS} scale={SCALE}
Epochs: {EPOCHS}  lr={LR}
Test accuracy: {acc:.4f}

## Entropy Cascade
| Layer | H(marginal) bits | cH(L|prev) bits | drop ratio | r_affine |
|-------|-----------------|-----------------|------------|----------|
"""
for i in range(N_LAYERS):
    summary += f"| L{i+1}   | {H_marginal[i]:.4f}          | {cH[i]:.4f}          | {drop_ratios[i]:.3f}       | {r_vals[i]:.4f}   |\n"

summary += f"""
## Correlation
Spearman(conditional_H_ratio, r_affine) = {rho:.4f}  p={pval:.4f}

## Verdict: {verdict.upper()}
{reason}
"""

with open("autoresearch/results/cycle_019_summary.md", "w") as f:
    f.write(summary)

print("Done.")
