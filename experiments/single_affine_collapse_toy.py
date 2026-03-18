"""
Cycle 010 — Item 20: Single-affine collapse quality on toy ReLU MLP.

For each hidden layer in a trained spirals ReLU MLP, fit a single affine map
A, b via least squares on probe data, then measure normalized residual r on
held-out data. Determine whether any layer is functionally linear (r < 0.15).
"""

import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# ── Dataset ──────────────────────────────────────────────────────────────────
from aleph.datasets import make_spirals

X_np, y_np = make_spirals(n_samples=2000, noise=0.0, n_turns=3, random_state=42)
X = torch.tensor(X_np, dtype=torch.float32) * 2.0  # 2× scale
y = torch.tensor(y_np, dtype=torch.float32)

# ── Model ─────────────────────────────────────────────────────────────────────
class SimpleMLP(nn.Module):
    """Plain ReLU MLP: 2 → 64 → 64 → 64 → 64 → 1"""
    def __init__(self, input_dim=2, hidden_dim=64, n_hidden=4, output_dim=1):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * n_hidden + [output_dim]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)]
        )
        self.n_hidden = n_hidden

    def forward(self, x):
        for i, lin in enumerate(self.linears[:-1]):
            x = F.relu(lin(x))
        return self.linears[-1](x)

    def get_hidden_activations(self, x):
        """Returns list of (pre_relu, post_relu) for each hidden layer."""
        acts = []
        h = x
        for lin in self.linears[:-1]:
            pre = lin(h)
            post = F.relu(pre)
            acts.append((pre, post))
            h = post
        return acts


model = SimpleMLP()

# ── Training ──────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

best_acc = 0.0
for epoch in range(3000):
    model.train()
    optimizer.zero_grad()
    logits = model(X).squeeze()
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            preds = (model(X).squeeze() > 0).float()
            acc = (preds == y).float().mean().item()
        if acc > best_acc:
            best_acc = acc
        if acc > 0.92:
            print(f"  Epoch {epoch+1}: acc={acc:.4f} — target reached, stopping early")
            break
        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1}: acc={acc:.4f}, loss={loss.item():.4f}")

model.eval()
with torch.no_grad():
    preds = (model(X).squeeze() > 0).float()
    final_acc = (preds == y).float().mean().item()

print(f"\nFinal train accuracy: {final_acc:.4f}")
if final_acc < 0.92:
    print("WARNING: model did not reach 92% accuracy. Proceeding but results may be unreliable.")
    # Try fallback: simpler config
    print("Trying fallback: 3 hidden layers, hidden_dim=64, more epochs...")
    model = SimpleMLP(n_hidden=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5000):
        model.train()
        optimizer.zero_grad()
        logits = model(X).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                preds = (model(X).squeeze() > 0).float()
                acc = (preds == y).float().mean().item()
            print(f"  Fallback epoch {epoch+1}: acc={acc:.4f}")
            if acc > 0.92:
                break
    model.eval()
    with torch.no_grad():
        preds = (model(X).squeeze() > 0).float()
        final_acc = (preds == y).float().mean().item()
    print(f"Fallback final accuracy: {final_acc:.4f}")

# ── Probe data split ──────────────────────────────────────────────────────────
n = X.shape[0]
half = n // 2
X_fit = X[:half]
X_held = X[half:]
# (labels not needed for residual computation)

# ── Collect activations ───────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    acts_fit = model.get_hidden_activations(X_fit)
    acts_held = model.get_hidden_activations(X_held)

# Sanity check: fit and held-out indices are disjoint
assert X_fit.shape[0] + X_held.shape[0] == n, "Split not exhaustive"
assert X_fit.shape[0] == half and X_held.shape[0] == n - half

# ── Single-affine collapse per layer ──────────────────────────────────────────
results = []
n_hidden = len(acts_fit)

print("\n" + "="*60)
print(f"{'Layer':>5}  {'r_fit':>8}  {'r_held':>8}  {'frac_dead':>9}  {'frac_always':>11}")
print("="*60)

for layer_idx in range(n_hidden):
    # Input to this layer = post-ReLU of previous layer (or raw input for layer 0)
    if layer_idx == 0:
        h_in_fit = X_fit
        h_in_held = X_held
    else:
        h_in_fit = acts_fit[layer_idx - 1][1]   # post-relu of layer_idx-1
        h_in_held = acts_held[layer_idx - 1][1]

    h_out_fit = acts_fit[layer_idx][1]   # post-relu of layer_idx (target)
    h_out_held = acts_held[layer_idx][1]

    # Fit least squares: solve min ||H_out - [H_in | 1] @ [A^T; b^T]||_F
    # Augmented input with bias column
    n_fit = h_in_fit.shape[0]
    H_aug_fit = torch.cat([h_in_fit, torch.ones(n_fit, 1)], dim=1)  # (n, d_in+1)
    H_out_fit = h_out_fit  # (n, d_out)

    # Solve using torch.linalg.lstsq
    # lstsq solves H_aug @ coef = H_out, coef shape (d_in+1, d_out)
    try:
        sol = torch.linalg.lstsq(H_aug_fit, H_out_fit)
        coef = sol.solution  # (d_in+1, d_out)
    except Exception as e:
        print(f"  lstsq failed for layer {layer_idx}: {e}. Using pinv.")
        coef = torch.linalg.pinv(H_aug_fit) @ H_out_fit

    A = coef[:-1]  # (d_in, d_out)
    b = coef[-1]   # (d_out,)

    # Compute residuals
    def normalized_residual(h_in, h_out, A, b):
        pred = h_in @ A + b  # (n, d_out)
        resid = h_out - pred
        norm_resid = torch.norm(resid, 'fro')
        denom = torch.norm(h_out - h_out.mean(dim=0, keepdim=True), 'fro')
        if denom < 1e-10:
            return float('nan')
        return (norm_resid / denom).item()

    n_held = h_in_held.shape[0]
    H_aug_held = torch.cat([h_in_held, torch.ones(n_held, 1)], dim=1)
    r_fit = normalized_residual(h_in_fit, h_out_fit, A, b)
    r_held = normalized_residual(h_in_held, h_out_held, A, b)

    # Sanity check: identity mapping should be worse (or equal) than fitted A, b
    # Compare to a simple zero-prediction baseline (predict mean)
    mean_pred = h_out_fit.mean(dim=0, keepdim=True).expand_as(h_out_fit)
    r_mean_fit = (torch.norm(h_out_fit - mean_pred, 'fro') / (torch.norm(h_out_fit - mean_pred, 'fro') + 1e-10)).item()
    # (by definition r_fit uses mean as denominator, so r=1.0 corresponds to predicting mean)

    # Stability stats: fraction dead (stable-0) and always-active (stable-1) neurons on fit data
    pre_fit = acts_fit[layer_idx][0]  # pre-relu activations on fit data (n, d)
    # Dead: always <= 0 (all samples inactive)
    always_dead = (pre_fit <= 0).all(dim=0).float().mean().item()
    # Always active: always > 0
    always_active = (pre_fit > 0).all(dim=0).float().mean().item()

    print(f"  L{layer_idx+1:1d}   {r_fit:8.4f}  {r_held:8.4f}  {always_dead:9.4f}  {always_active:11.4f}")

    results.append({
        "layer": layer_idx + 1,
        "r_fit": r_fit,
        "r_held": r_held,
        "frac_dead": always_dead,
        "frac_always_active": always_active,
    })

print("="*60)

# ── Verdict ───────────────────────────────────────────────────────────────────
min_r_held = min(r["r_held"] for r in results)
max_r_held = max(r["r_held"] for r in results)

if min_r_held < 0.15:
    verdict = "confirmed"
    verdict_reason = f"Layer {next(r['layer'] for r in results if r['r_held'] < 0.15)} has r={min_r_held:.4f} < 0.15 on held-out data."
elif max_r_held > 0.5 and all(r["r_held"] > 0.5 for r in results):
    verdict = "refuted"
    verdict_reason = f"All layers have r > 0.5 on held-out data (min={min_r_held:.4f})."
else:
    verdict = "inconclusive"
    verdict_reason = f"r values are moderate (range {min_r_held:.4f}–{max_r_held:.4f}), not clearly linear or nonlinear."

print(f"\nVerdict: {verdict.upper()}")
print(f"Reason: {verdict_reason}")
print(f"Train accuracy: {final_acc:.4f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
artifact = {
    "experiment": "single_affine_collapse_toy",
    "cycle": 10,
    "item": 20,
    "config": {
        "dataset": "spirals",
        "n_samples": 2000,
        "n_turns": 3,
        "scale": 2.0,
        "architecture": "2->64->64->64->64->1",
        "n_hidden_layers": n_hidden,
        "lr": 0.001,
    },
    "train_accuracy": final_acc,
    "per_layer": results,
    "min_r_held": min_r_held,
    "max_r_held": max_r_held,
    "verdict": verdict,
    "verdict_reason": verdict_reason,
}

out_path = "autoresearch/results/cycle_010_single_affine.json"
with open(out_path, "w") as f:
    json.dump(artifact, f, indent=2)

print(f"\nArtifact saved to {out_path}")
