# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: quant-aware-training
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Canonical Error Correction for Quantized Neural Networks
#
# **Question**: When we quantize a neural network's weights, error accumulates
# through layers. Can we correct it? How much? Where should we focus?
#
# **Setup**: A trained MLP (2→8→8→1) classifying moons, with 4-bit weight
# quantization. Biases stay float (they cancel in the error math).
#
# **Approach**: We run the float and quantized networks side by side, measure
# where errors come from (this layer vs propagated from earlier layers), and
# apply an oracle correction that knows the float activations. This tells us
# the theoretical ceiling — what's fixable vs what's not.
#
# **Canonical space**: All errors are mapped back to input space via $T_L^+ =
# (W_L \cdots W_1)^+$. This gives a consistent 2D coordinate system for
# comparing errors across layers of different widths (2D input, 8D hidden, 1D
# output). The tradeoff: $T_L^+$ is a linear approximation that ignores ReLU,
# so it's exact for the linear error component and approximate for the rest.

# %%
import numpy as np  # only for matplotlib mesh grids
import matplotlib.pyplot as plt
from typing import List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons

from aleph.qgeom import (
    CanonicalSpaceTracker,
    ForwardTrace,
    ReLUDisagreementTracker,
    error_attribution,
)

# --- Config ---
BITS = 4
DELTA = 1.0 / (2 ** (BITS - 1))
SEED = 42
PLOT_RANGE = 2.5
GRID_N = 50
QUIVER_N = 12

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Quantization: {BITS}-bit, delta={DELTA:.6f}")

# %% [markdown]
# ## Setup: Train and Quantize

# %%
def make_mlp(hidden_dim=8, depth=8, input_dim=2, output_dim=1):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def train_model(model, X, y, epochs=5000, lr=0.01):
    """Train the supplied `model` on (X, y) and return (model, accuracy)."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_t)) > 0.5).float()
        acc = (preds == y_t).float().mean().item()
    return model, acc


def extract_weights(model, delta=DELTA):
    weights, biases = [], []
    for module in model:
        if isinstance(module, nn.Linear):
            weights.append(module.weight.detach().clone())
            biases.append(module.bias.detach().clone())
    weights_q = [torch.round(W / delta) * delta for W in weights]
    return weights, biases, weights_q


X_data, y_data = make_moons(n_samples=500, noise=0.15, random_state=SEED)
# create model externally and pass it to the trainer for reproducibility
model = make_mlp(hidden_dim=8, depth=8)
model, acc = train_model(model, X_data, y_data)
weights, biases, weights_q = extract_weights(model)

# adapt the printed architecture to the actual trained model
arch_dims = [weights[0].shape[1]] + [int(W.shape[0]) for W in weights]
arch_str = ' -> '.join(str(d) for d in arch_dims)
print(f"Architecture: {arch_str} (ReLU between hidden layers)")
print(f"Float accuracy: {acc:.1%}")
print(f"\nPer-layer quantization error (||W_q - W||_F):")
for i, (W, Wq) in enumerate(zip(weights, weights_q)):
    E = Wq - W
    print(f"  Layer {i}: {tuple(W.shape)} -> ||E||_F = {torch.linalg.norm(E, 'fro'):.4f}")

# %% [markdown]
# ## Core Framework
#
# The analysis tools live in `aleph.qgeom.canonical`:
#
# - **CanonicalSpaceTracker**: computes $T_L = W_L \cdots W_1$ and maps
#   errors to input space via pseudoinverse $T_L^+$
# - **ReLUDisagreementTracker**: finds where $\text{sign}(z_L^{\text{float}})
#   \neq \text{sign}(z_L^{\text{quant}})$
# - **error_attribution**: decomposes each layer's error into local vs propagated
#
# The notebook provides the forward pass and the oracle corrector:
#
# - **forward_pass**: runs the network, collects pre/post-activation traces
# - **PerfectCorrection**: applies $C_L = -E_L \hat{a}_{L-1} - W_L
#   \varepsilon_{L-1}$ at each layer (oracle: needs float activations)

# %%
@torch.no_grad()
def forward_pass(x, weights, biases):
    """Run forward pass, collecting pre- and post-activation tensors."""
    a = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    pre_acts, post_acts = [], []
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = F.linear(a, W, b)
        a = z if i == len(weights) - 1 else F.relu(z)
        pre_acts.append(z)
        post_acts.append(a)
    return ForwardTrace(pre_acts=pre_acts, post_acts=post_acts)


class PerfectCorrection:
    """Oracle correction: C_L = -E_L @ â_{L-1} - W_L @ ε_{L-1}.

    Requires float activations (oracle). Biases are in the forward pass
    but cancel in the correction formula.
    """

    def __init__(self, weights, weights_q, biases):
        self.weights, self.weights_q, self.biases = weights, weights_q, biases
        self.errors = [Wq - W for W, Wq in zip(weights, weights_q)]

    @torch.no_grad()
    def run(self, x, float_trace, correct_at=None):
        if correct_at is None:
            correct_at = set(range(len(self.weights)))
        a = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        epsilon = torch.zeros_like(a)
        pre_acts, post_acts, corrections = [], [], []
        for i in range(len(self.weights)):
            is_last = (i == len(self.weights) - 1)
            E, W, Wq, b = self.errors[i], self.weights[i], self.weights_q[i], self.biases[i]
            if i in correct_at:
                C = -F.linear(a, E) - F.linear(epsilon, W)
                z = F.linear(a, Wq, b) + C
            else:
                C = torch.zeros_like(F.linear(a, Wq))
                z = F.linear(a, Wq, b)
            a_new = z if is_last else F.relu(z)
            epsilon = a_new - float_trace.post_acts[i]
            pre_acts.append(z)
            post_acts.append(a_new)
            corrections.append(C)
            a = a_new
        return ForwardTrace(pre_acts, post_acts), corrections

# %% [markdown]
# ---
# ## Experiment 1: Error Compounds Through Layers
#
# The first question: when we quantize weights, how does error behave as it
# flows through the network?
#
# At each layer, the pre-ReLU error is:
# $$\hat{z}_L - z_L = \underbrace{E_L \cdot \hat{a}_{L-1}}_{\text{local: this
# layer's bad weights}} + \underbrace{W_L \cdot \varepsilon_{L-1}}_{\text{propagated:
# previous errors amplified forward}}$$
#
# **Local error** is the direct effect of quantizing layer $L$'s weights.
# **Propagated error** is everything that went wrong before, multiplied by
# $W_L$. In a trained network where most neurons are active (unlike random 2D
# weights where ReLU kills half the signal), propagated error grows.

# %%
tracker = CanonicalSpaceTracker(weights)
X_t = torch.tensor(X_data, dtype=torch.float32)
float_trace = forward_pass(X_t, weights, biases)
quant_trace = forward_pass(X_t, weights_q, biases)
attrib = error_attribution(X_t, weights, weights_q, float_trace, quant_trace, tracker)

n_layers = len(weights)
layers_idx = [r['layer'] for r in attrib]
local_out = [r['local_output'].norm(dim=-1).mean().item() for r in attrib]
prop_out = [r['propagated_output'].norm(dim=-1).mean().item() for r in attrib]
total_out = [r['total_output'].norm(dim=-1).mean().item() for r in attrib]

fig, ax = plt.subplots(figsize=(7, 5))
bars_local = ax.bar(layers_idx, local_out, label='Local (this layer)', color='steelblue')
bars_prop = ax.bar(layers_idx, prop_out, bottom=local_out, label='Propagated (previous layers)', color='coral')

# Annotate percentages
for i, (lo, po) in enumerate(zip(local_out, prop_out)):
    total = lo + po
    if total > 0:
        ax.text(i, total + 0.02, f'{po/total*100:.0f}% propagated',
                ha='center', fontsize=9, color='coral')

ax.set_xlabel('Layer')
ax.set_ylabel('Mean pre-ReLU error (output space)')
ax.set_title(f'Error Decomposition: Local vs Propagated ({BITS}-bit)')
ax.set_xticks(layers_idx)
ax.set_xticklabels([f'L{i}\n{tuple(w.shape)}' for i, w in enumerate(weights)])
ax.legend()
plt.tight_layout()
plt.savefig('plots/canonical_error_attribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("Layer-by-layer breakdown:")
for i, (lo, po, to) in enumerate(zip(local_out, prop_out, total_out)):
    total = lo + po
    pct = po / total * 100 if total > 0 else 0
    print(f"  L{i} ({tuple(weights[i].shape)}): local={lo:.4f}, propagated={po:.4f}, "
          f"total={to:.4f} ({pct:.0f}% propagated)")

# %% [markdown]
# **What this shows**: Layer 0 has zero propagated error (it's the first layer).
# By layer 2, propagated error is roughly half the total — earlier quantization
# mistakes are being amplified forward through the network.
#
# The total error grows from ~0.09 at layer 0 to ~1.1 at the output. This is
# the compounding problem that correction aims to solve.

# %% [markdown]
# ### Where in input space is the error?

# %%
grid_1d = np.linspace(-PLOT_RANGE, PLOT_RANGE, GRID_N)
xx, yy = np.meshgrid(grid_1d, grid_1d)
grid_t = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)

float_grid = forward_pass(grid_t, weights, biases)
quant_grid = forward_pass(grid_t, weights_q, biases)

fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4.5))
for i in range(n_layers):
    ax = axes[i] if n_layers > 1 else axes
    err = (quant_grid.post_acts[i] - float_grid.post_acts[i]).norm(dim=-1).numpy()
    im = ax.contourf(xx, yy, err.reshape(GRID_N, GRID_N), levels=20, cmap='viridis')
    plt.colorbar(im, ax=ax, label='||â - a||')
    ax.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1],
               c='white', s=6, alpha=0.3, edgecolors='none')
    ax.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1],
               c='white', s=6, alpha=0.3, edgecolors='none')
    ax.set_title(f'Layer {i} ({tuple(weights[i].shape)})\npost-activation error')
    ax.set_xlim(-PLOT_RANGE, PLOT_RANGE)
    ax.set_ylim(-PLOT_RANGE, PLOT_RANGE)
    ax.set_aspect('equal')

plt.suptitle(f'Where is the error (canonical space)? (white dots = training data, {BITS}-bit)', y=1.02)
plt.tight_layout()
plt.savefig('plots/canonical_error_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# Error is not uniform across input space — it concentrates in regions where
# the network has large activations (more to quantize = more error).

# %% [markdown]
# ---
# ## Experiment 2: Perfect Correction Eliminates Error
#
# **The correction formula**: at each layer, subtract the quantization error:
# $$C_L = -E_L \cdot \hat{a}_{L-1} - W_L \cdot \varepsilon_{L-1}$$
#
# The first term undoes this layer's weight quantization error. The second term
# undoes the propagated error from previous layers. Together they recover the
# float pre-activation exactly: $\hat{z}_L + C_L = z_L$.
#
# This is an **oracle** — it needs the float activations to compute
# $\varepsilon_{L-1}$. It shows the theoretical ceiling: what's achievable
# if we had perfect information.

# %%
corrector = PerfectCorrection(weights, weights_q, biases)
corr_trace, _ = corrector.run(X_t, float_trace)

# Per-layer error comparison
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(n_layers)
width = 0.35

uncorr_errs = [(quant_trace.post_acts[i] - float_trace.post_acts[i]).norm(dim=-1).mean().item()
               for i in range(n_layers)]
corr_errs = [(corr_trace.post_acts[i] - float_trace.post_acts[i]).norm(dim=-1).mean().item()
             for i in range(n_layers)]

bars1 = ax.bar(x_pos - width/2, uncorr_errs, width, label='Uncorrected', color='coral')
bars2 = ax.bar(x_pos + width/2, corr_errs, width, label='Corrected', color='steelblue')

# Annotate corrected values (they're ~0)
for i, v in enumerate(corr_errs):
    ax.text(i + width/2, max(uncorr_errs) * 0.02, f'{v:.1e}',
            ha='center', va='bottom', fontsize=8, color='steelblue')

ax.set_xlabel('Layer')
ax.set_ylabel('Mean ||â_L - a_L|| (post-activation error)')
ax.set_title('Uncorrected vs Corrected Error at Each Layer')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'L{i}\n{tuple(weights[i].shape)}' for i in range(n_layers)])
ax.legend()
plt.tight_layout()
plt.savefig('plots/canonical_correction_verification.png', dpi=150, bbox_inches='tight')
plt.show()

print("Per-layer residual after correction:")
for i, (ue, ce) in enumerate(zip(uncorr_errs, corr_errs)):
    print(f"  L{i}: uncorrected={ue:.4f}, corrected={ce:.2e} (ratio: {ce/ue:.1e})")

# %% [markdown]
# The corrected error is ~$10^{-7}$ at every layer — float32 machine precision.
# Perfect correction recovers the float network exactly, confirming the math.
#
# ### Which layers matter most to correct?
#
# If we can only correct at some layers (because correction is expensive),
# where should we spend the budget?

# %%
strategies = [
    (set(range(n_layers)), "All layers"),
    ({0}, "Layer 0 only (first)"),
    ({1}, "Layer 1 only (middle)"),
    ({n_layers - 1}, f"Layer {n_layers-1} only (output)"),
    ({0, 1}, "Layers 0+1 (both hidden)"),
    (set(), "No correction"),
]

strat_labels = []
strat_errors = []
for correct_at, label in strategies:
    res_trace, _ = corrector.run(X_t, float_trace, correct_at=correct_at)
    err = (res_trace.post_acts[-1] - float_trace.post_acts[-1]).norm(dim=-1).mean().item()
    strat_labels.append(label)
    strat_errors.append(err)

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['steelblue' if e < 1e-10 else 'coral' for e in strat_errors]
bars = ax.barh(range(len(strategies)), strat_errors, color=colors)
ax.set_yticks(range(len(strategies)))
ax.set_yticklabels(strat_labels)
ax.set_xlabel('Mean output error')
ax.set_title('Partial Correction: Which Layers Matter?')
ax.invert_yaxis()

for i, v in enumerate(strat_errors):
    ax.text(v + max(strat_errors) * 0.02, i, f'{v:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('plots/canonical_partial_correction.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# **Key finding**: correcting the output layer alone achieves near-zero error.
# This makes sense — the output layer (1×8) is the bottleneck. Its correction
# $C_2$ has enough degrees of freedom to compensate for all upstream error
# accumulated in the 8D hidden representation.
#
# Correcting only layer 0 helps (reduces error from 1.1 to ~0.7) but can't
# fix errors introduced at layers 1 and 2. Correcting both hidden layers
# (0+1) still leaves the output layer's own quantization error.
#
# **Implication for the project**: correction placement depends on architecture.
# At a bottleneck (wide → narrow), a single correction can absorb a lot.
# Between same-width layers, you need to correct more often.

# %% [markdown]
# ---
# ## Experiment 3: ReLU Disagreement — The Theoretical Limit
#
# Where the quantized and float networks make different ReLU decisions
# ($\text{sign}(z) \neq \text{sign}(\hat{z})$), information is permanently
# lost. This is the only source of error that perfect correction can't fix
# in the general case (though here it does fix it because we correct at every
# layer, so pre-activations match exactly).
#
# Understanding where disagreement occurs tells us where a *practical*
# (non-oracle) correction would struggle.

# %%
relu_tracker = ReLUDisagreementTracker(float_grid, quant_grid)

n_relu_layers = len(relu_tracker.disagreements)
fig, axes = plt.subplots(1, n_relu_layers + 1, figsize=(5 * (n_relu_layers + 1), 4.5))

any_disagree = torch.zeros(grid_t.shape[0])
for i in range(n_relu_layers):
    ax = axes[i]
    mask = relu_tracker.any_disagreement(i).float().numpy()
    ax.contourf(xx, yy, mask.reshape(GRID_N, GRID_N),
                levels=[-0.5, 0.5, 1.5], colors=['white', 'coral'], alpha=0.7)
    ax.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1],
               c='#1f77b4', s=8, alpha=0.3, edgecolors='none')
    ax.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1],
               c='#d62728', s=8, alpha=0.3, edgecolors='none')
    frac = relu_tracker.fractions[i]
    ax.set_title(f'Layer {i}: {frac*100:.1f}% disagreement')
    ax.set_xlim(-PLOT_RANGE, PLOT_RANGE)
    ax.set_ylim(-PLOT_RANGE, PLOT_RANGE)
    ax.set_aspect('equal')
    any_disagree += relu_tracker.any_disagreement(i).float()

ax = axes[-1]
im = ax.contourf(xx, yy, any_disagree.numpy().reshape(GRID_N, GRID_N),
                 levels=np.arange(n_relu_layers + 2) - 0.5, cmap='Reds')
plt.colorbar(im, ax=ax, label='# layers disagreeing')
ax.scatter(X_data[:, 0], X_data[:, 1], c='gray', s=6, alpha=0.3, edgecolors='none')
ax.set_title('Combined (any layer)')
ax.set_xlim(-PLOT_RANGE, PLOT_RANGE)
ax.set_ylim(-PLOT_RANGE, PLOT_RANGE)
ax.set_aspect('equal')

plt.suptitle('ReLU Disagreement: Where Quantization Flips Neuron Decisions', y=1.02)
plt.tight_layout()
plt.savefig('plots/canonical_relu_disagreement.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# Disagreement is low (~1-3% of activations) at 4-bit and spatially structured.
# The red regions are where some neuron is close to zero in the float network,
# so even a small quantization perturbation flips it. These regions would be
# the hard cases for a learned (non-oracle) correction.

# %% [markdown]
# ---
# ## Experiment 4: Canonical Space and Geometric Properties
#
# The canonical space maps all errors to the 2D input space via $T_L^+$,
# giving a single coordinate system for comparison. This is useful because
# output space dimensions change between layers (2 → 8 → 8 → 1).
#
# The quiver plot below shows, for each point in input space, the direction
# and magnitude of the canonical error at each layer. If errors at different
# layers point in the same direction, they compound. If they point in
# different directions, they partially cancel.

# %%
quiver_1d = np.linspace(-PLOT_RANGE, PLOT_RANGE, QUIVER_N)
qxx, qyy = np.meshgrid(quiver_1d, quiver_1d)
quiver_t = torch.tensor(np.stack([qxx.ravel(), qyy.ravel()], axis=1), dtype=torch.float32)
q_float = forward_pass(quiver_t, weights, biases)
q_quant = forward_pass(quiver_t, weights_q, biases)
quiver_attrib = error_attribution(quiver_t, weights, weights_q, q_float, q_quant, tracker)

fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))
all_norms = [r['total_canonical'].norm(dim=-1).max().item() for r in quiver_attrib]
max_norm = max(all_norms) if all_norms else 1.0

for i, r in enumerate(quiver_attrib):
    ax = axes[i] if n_layers > 1 else axes
    err = r['total_canonical'].numpy()
    norms = np.linalg.norm(err, axis=-1)
    ax.quiver(qxx.ravel(), qyy.ravel(),
              err[:, 0], err[:, 1], norms,
              cmap='coolwarm', scale=max_norm * QUIVER_N * 0.8, alpha=0.8)
    ax.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1],
               c='#1f77b4', s=8, alpha=0.15, edgecolors='none')
    ax.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1],
               c='#d62728', s=8, alpha=0.15, edgecolors='none')
    ax.set_xlim(-PLOT_RANGE, PLOT_RANGE)
    ax.set_ylim(-PLOT_RANGE, PLOT_RANGE)
    ax.set_aspect('equal')
    ax.set_title(f'Layer {i} canonical error')

plt.suptitle('Error Vectors Mapped to Input Space (canonical)', y=1.02)
plt.tight_layout()
plt.savefig('plots/canonical_error_quiver.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Geometric metrics table
def compute_geometric_metrics(weights, weights_q, tracker):
    metrics = []
    for i, (W, Wq) in enumerate(zip(weights, weights_q)):
        E = Wq - W
        metrics.append({
            'layer': i, 'shape': tuple(W.shape),
            'spectral_E': torch.linalg.norm(E, ord=2).item(),
            'frobenius_E': torch.linalg.norm(E, 'fro').item(),
            'spectral_W': torch.linalg.norm(W, ord=2).item(),
            'amplification': tracker.cumulative_amplification(i),
            'cond_T': torch.linalg.cond(tracker.cumulative_transform(i)).item(),
        })
    return metrics


def error_pca(errors):
    centered = errors - errors.mean(dim=0)
    if centered.norm() < 1e-15:
        return torch.ones(errors.shape[-1]) / errors.shape[-1] ** 0.5, 0.0
    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    explained = S ** 2 / (S ** 2).sum()
    return Vh[0], explained[0].item()


geo = compute_geometric_metrics(weights, weights_q, tracker)

print("=== Geometric Metrics ===\n")
print(f"{'Layer':<6s} {'Shape':<10s} {'||E||_2':<9s} {'||W||_2':<9s} "
      f"{'||T_L||_2':<10s} {'cond(T_L)':<10s}")
print("-" * 58)
for m in geo:
    print(f"{m['layer']:<6d} {str(m['shape']):<10s} {m['spectral_E']:<9.4f} "
          f"{m['spectral_W']:<9.4f} {m['amplification']:<10.4f} {m['cond_T']:<10.4f}")

print("\nCanonical error PCA (what direction does error point in input space?):")
for r in attrib:
    d, v = error_pca(r['total_canonical'])
    print(f"  L{r['layer']}: [{d[0].item():+.3f}, {d[1].item():+.3f}], {v*100:.0f}% variance explained")

# %% [markdown]
# **Reading the metrics**:
#
# - $\|E_L\|_2$: how much error this layer's quantization introduces per unit
#   activation. Larger = worse quantization at this layer.
# - $\|T_L\|_2$: how much the network amplifies signals up to layer $L$.
#   Larger = errors at layer $L$ look smaller in canonical space (they've been
#   amplified, so dividing by $T_L$ normalizes them down).
# - $\text{cond}(T_L)$: how numerically stable the canonical mapping is.
#   Large condition number means the pseudoinverse is unreliable.
# - **PCA**: errors are highly directional (>99% in one component). The
#   quantization grid imposes a preferred direction of error in input space.

# %% [markdown]
# ---
# ## Experiment 5: How Does Depth Change the Picture?
#
# Train networks of different depths and see how the local/propagated
# balance shifts.

# %%
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

for col, depth in enumerate([2, 3, 4, 5]):
    torch.manual_seed(SEED)
    # create the model with the desired depth, then train it (pass model in)
    m = make_mlp(hidden_dim=8, depth=depth)
    m, a = train_model(m, X_data, y_data)
    w, bs, wq = extract_weights(m)
    t = CanonicalSpaceTracker(w)
    ft = forward_pass(X_t, w, bs)
    qt = forward_pass(X_t, wq, bs)
    att = error_attribution(X_t, w, wq, ft, qt, t)

    idx = [r['layer'] for r in att]
    lo = [r['local_output'].norm(dim=-1).mean().item() for r in att]
    po = [r['propagated_output'].norm(dim=-1).mean().item() for r in att]

    ax = axes[col]
    ax.bar(idx, lo, label='Local', color='steelblue')
    ax.bar(idx, po, bottom=lo, label='Propagated', color='coral')
    ax.set_xlabel('Layer')
    ax.set_title(f'depth={depth} ({depth+1} layers total)')
    ax.set_xticks(idx)
    if col == 0:
        ax.set_ylabel('Mean pre-ReLU error')
    if col == 3:
        ax.legend()

plt.suptitle(f'Error Decomposition vs Network Depth ({BITS}-bit)', y=1.02)
plt.tight_layout()
plt.savefig('plots/canonical_error_vs_depth.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# With more layers, propagated error increasingly dominates. By depth=5, the
# last few layers are almost entirely propagated error — their own local
# quantization error is a small fraction of the total.
#
# **Implication**: in deep networks, fixing early layers has outsized impact
# because their errors compound through all subsequent layers. Late-layer
# corrections can also work (as we saw with the output layer trick) but only
# if the correction has enough capacity to absorb the accumulated error.

# %% [markdown]
# ---
# ## Summary
#
# | Finding | Implication |
# |---------|-------------|
# | Error compounds through layers (propagated grows from 0% to ~50%) | Early correction has outsized impact |
# | Perfect oracle correction gives zero residual at every layer | The correction formula $C_L = -E_L \hat{a} - W_L \varepsilon$ is exact |
# | Output-layer-only correction achieves near-zero error | Bottleneck layers (wide→narrow) can absorb upstream error |
# | ReLU disagreement is ~1-3% and spatially structured | Practical corrections will struggle near decision boundaries |
# | Canonical errors are highly directional (>99% in 1 PC) | Error has geometric structure, not random noise |
#
# **Next step** (efficiency phase): replace the oracle with a learned correction
# that estimates $\varepsilon_{L-1}$ from the quantized activations alone. The
# canonical space framework tells us where to focus that learned correction.

# %% [markdown]
#
