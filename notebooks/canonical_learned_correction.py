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
# # Geometric Error Decomposition: Metric vs Topological Distortion
#
# **Question**: How much of quantization error is linearly correctable, and
# what is the geometric structure of the remainder?
#
# **Geometric setup**: Each row of a weight matrix $W_L$ defines a hyperplane
# in activation space, and ReLU partitions the space at that hyperplane into
# two half-spaces (active vs inactive). Quantization perturbs $W \to W_q$,
# which shifts these hyperplanes. Two things can happen:
#
# 1. **Metric distortion** — the input stays in the same half-space. The
#    transform rotates and stretches differently, but the topology of the
#    representation is preserved. This is a linear perturbation with a
#    linear (or low-rank) correction.
#
# 2. **Topological distortion** — the hyperplane shifts enough that an input
#    crosses the boundary. A dimension carrying manifold structure collapses
#    to zero (or a zero inflates to a manifold). This is a qualitative change
#    in the representation geometry. No linear correction can undo it.
#
# This notebook measures the per-layer budget: what fraction of the error is
# metric (cheap to fix) vs topological (needs capacity or is irreducible).
# The result establishes the ceiling for any linear/low-rank correction and
# provides the geometric explanation for why linear corrections empirically
# outperform MLPs (see `docs/quantization_correction_system.md`).
#
# **Relation to prior work**: The canonical error notebook
# (`canonical_error_correction.py`) decomposes error into local vs propagated
# and counts ReLU disagreement rates, but treats all error as one thing to
# correct. The old `error_tracking_viz.py` notebook decomposes by numerical
# source (weight rounding, activation rounding, clipping, ReLU interaction)
# in raw activation space — an accounting exercise with no geometric framing.
# This notebook splits by *what kind of geometric distortion* the error
# represents, in canonical space, and directly connects to correction design.

# %%
import numpy as np  # only for matplotlib mesh grids
import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from aleph.datasets import make_spirals
from aleph.qgeom import (
    CanonicalSpaceTracker,
    ForwardTrace,
    ReLUDisagreementTracker,
    error_attribution,
)

# --- Config (matches canonical_error_correction.py) ---
BITS = 4
DELTA = 1.0 / (2 ** (BITS - 1))
SEED = 42
HIDDEN_DIM = 32
DEPTH = 12
LR = 0.001
N_SAMPLES = 2000

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Config: {BITS}-bit, delta={DELTA:.6f}, {HIDDEN_DIM}×{DEPTH}")

# %% [markdown]
# ## Train and quantize

# %%
def make_mlp(hidden_dim=HIDDEN_DIM, depth=DEPTH, input_dim=2, output_dim=1):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def train_model(model, X, y, epochs=5000, lr=LR):
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


X_data, y_data = make_spirals(n_samples=N_SAMPLES, noise=0.5, n_turns=3, random_state=SEED)
X_data *= 2.0

model = make_mlp()
model, acc = train_model(model, X_data, y_data)
weights, biases, weights_q = extract_weights(model)

assert acc > 0.85, f"Model only reached {acc:.1%} — undertrained, analysis is meaningless"
print(f"Float accuracy: {acc:.1%}")

# %% [markdown]
# ## Forward traces
#
# Run both float and quantized networks, collecting pre/post-activation
# tensors at every layer.

# %%
@torch.no_grad()
def forward_pass(x, weights, biases):
    a = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    pre_acts, post_acts = [], []
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = F.linear(a, W, b)
        a = z if i == len(weights) - 1 else F.relu(z)
        pre_acts.append(z)
        post_acts.append(a)
    return ForwardTrace(pre_acts=pre_acts, post_acts=post_acts)


X_t = torch.tensor(X_data, dtype=torch.float32)
float_trace = forward_pass(X_t, weights, biases)
quant_trace = forward_pass(X_t, weights_q, biases)

tracker = CanonicalSpaceTracker(weights)
relu_tracker = ReLUDisagreementTracker(float_trace, quant_trace)

# %% [markdown]
# ## Experiment 3: Metric vs topological error decomposition
#
# At each hidden layer, every neuron $i$ defines a hyperplane
# $\{x : w_i \cdot x + b_i = 0\}$ that partitions the space into two
# half-spaces. The ReLU keeps the active half-space and collapses the
# inactive half-space to zero.
#
# When quantization shifts $w_i \to w_{q,i}$, the hyperplane moves. For most
# inputs, they stay in the same half-space — the error is purely from the
# metric distortion of the linear transform ($E_L$ applied to the
# activation). For inputs near the boundary, the hyperplane shift pushes them
# across — a topological change that no linear correction can undo.
#
# We use the ReLU disagreement mask to separate these exactly:
#
# - **Agree mask** $M^{\text{agree}}_L$: neurons where
#   $\text{sign}(z^{\text{float}}) = \text{sign}(z^{\text{quant}})$.
#   The error on these neurons is the metric distortion component.
#
# - **Disagree mask** $M^{\text{disagree}}_L$: neurons where the signs
#   differ. The error on these neurons is the topological distortion
#   component.
#
# The total post-activation error decomposes exactly:
# $\varepsilon^{\text{post}}_L = \varepsilon^{\text{metric}}_L + \varepsilon^{\text{topo}}_L$

# %%
@torch.no_grad()
def geometric_error_decomposition(float_trace, quant_trace, weights, weights_q):
    """Decompose per-layer error into metric vs topological distortion.

    For each hidden layer:
    - metric: error on neurons where ReLU decision is preserved (same half-space)
    - topological: error on neurons where ReLU decision flipped (hyperplane crossing)

    Returns list of dicts, one per hidden layer (excludes output layer).
    """
    n_hidden = len(weights) - 1  # exclude output layer
    results = []

    for i in range(n_hidden):
        z_float = float_trace.pre_acts[i]   # (N, H)
        z_quant = quant_trace.pre_acts[i]    # (N, H)
        a_float = float_trace.post_acts[i]   # (N, H), after ReLU
        a_quant = quant_trace.post_acts[i]   # (N, H), after ReLU

        # Post-activation error (what actually propagates to next layer)
        eps_post = a_quant - a_float  # (N, H)

        # ReLU agreement mask: same half-space
        agree = (z_float > 0) == (z_quant > 0)  # (N, H)
        disagree = ~agree

        # Metric distortion: error on agreeing neurons only
        eps_metric = eps_post * agree.float()   # (N, H)
        # Topological distortion: error on disagreeing neurons only
        eps_topo = eps_post * disagree.float()  # (N, H)

        # Per-sample norms
        norm_total = torch.linalg.norm(eps_post, dim=1)      # (N,)
        norm_metric = torch.linalg.norm(eps_metric, dim=1)    # (N,)
        norm_topo = torch.linalg.norm(eps_topo, dim=1)        # (N,)

        # Fraction of neurons disagreeing (per sample, then averaged)
        disagree_frac = disagree.float().mean(dim=1)  # (N,)

        # Spectral structure of E_L restricted to agreeing subspace.
        # For each sample, the agreeing neurons are different, so we compute
        # the spectral properties of the *average* metric error.
        E = weights_q[i] - weights[i]  # (H, H_prev)

        # Variance explained by the metric error across the dataset:
        # what fraction of total ||eps||^2 is metric?
        total_sq = (norm_total ** 2).mean()
        metric_sq = (norm_metric ** 2).mean()
        topo_sq = (norm_topo ** 2).mean()
        metric_frac = (metric_sq / total_sq).item() if total_sq > 0 else 0.0
        topo_frac = (topo_sq / total_sq).item() if total_sq > 0 else 0.0

        # SVD of the metric error matrix (N, H) — how many dimensions does
        # the linearly correctable part actually use?
        U, S, Vh = torch.linalg.svd(eps_metric, full_matrices=False)
        cumvar = torch.cumsum(S ** 2, dim=0)
        total_var = cumvar[-1]
        if total_var > 0:
            explained = cumvar / total_var
            rank_90 = int((explained < 0.90).sum().item()) + 1
            rank_95 = int((explained < 0.95).sum().item()) + 1
            rank_99 = int((explained < 0.99).sum().item()) + 1
        else:
            rank_90 = rank_95 = rank_99 = 0

        # Information content of disagreeing neurons:
        # how much variance did the flipped neurons carry in the float network?
        float_var_per_neuron = a_float.var(dim=0)  # (H,)
        # Average the variance of neurons that disagree (across all samples)
        disagree_ever = disagree.any(dim=0)  # (H,) — neuron ever disagrees
        if disagree_ever.any():
            var_flipped = float_var_per_neuron[disagree_ever].mean().item()
            var_all = float_var_per_neuron.mean().item()
            var_ratio = var_flipped / var_all if var_all > 0 else 0.0
        else:
            var_flipped = 0.0
            var_all = float_var_per_neuron.mean().item()
            var_ratio = 0.0

        results.append({
            "layer": i,
            "eps_metric": eps_metric,
            "eps_topo": eps_topo,
            "eps_total": eps_post,
            "metric_frac": metric_frac,
            "topo_frac": topo_frac,
            "disagree_frac_mean": disagree_frac.mean().item(),
            "norm_total_mean": norm_total.mean().item(),
            "norm_metric_mean": norm_metric.mean().item(),
            "norm_topo_mean": norm_topo.mean().item(),
            "metric_svd_S": S,
            "metric_rank_90": rank_90,
            "metric_rank_95": rank_95,
            "metric_rank_99": rank_99,
            "n_neurons_disagree_ever": int(disagree_ever.sum().item()),
            "var_flipped_neurons": var_flipped,
            "var_all_neurons": var_all,
            "var_ratio": var_ratio,
            "E_spectral_norm": torch.linalg.norm(E, ord=2).item(),
            "E_frobenius_norm": torch.linalg.norm(E, ord='fro').item(),
        })

    return results


decomp = geometric_error_decomposition(float_trace, quant_trace, weights, weights_q)

# %% [markdown]
# ### Summary table

# %%
print(f"{'Layer':>5}  {'||err||':>7}  {'metric%':>8}  {'topo%':>7}  "
      f"{'disagree%':>9}  {'rank90':>6}  {'rank95':>6}  {'rank99':>6}  "
      f"{'var_ratio':>9}")
print("-" * 85)
for d in decomp:
    print(f"  L{d['layer']:<3}  {d['norm_total_mean']:7.3f}  "
          f"{d['metric_frac']:7.1%}  {d['topo_frac']:6.1%}  "
          f"{d['disagree_frac_mean']:8.1%}  "
          f"{d['metric_rank_90']:6d}  {d['metric_rank_95']:6d}  {d['metric_rank_99']:6d}  "
          f"{d['var_ratio']:9.3f}")

# %% [markdown]
# ### Interpretation
#
# - **metric%**: fraction of $||\varepsilon||^2$ from neurons in the same
#   half-space. This is the ceiling for any linear correction at this layer.
# - **topo%**: fraction from hyperplane crossings. Irreducible by linear
#   methods; needs nonlinear correction or is genuinely lost.
# - **rank90/95/99**: SVD rank of the metric error matrix needed to capture
#   that percentage of variance. Low rank = cheap linear correction.
# - **var_ratio**: how much variance the flipped neurons carried in the float
#   network, relative to all neurons. High ratio = the flips destroyed
#   important information. Low ratio = the flipped neurons were near-zero
#   anyway (cheap to lose).

# %% [markdown]
# ## Visualizing the decomposition

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

layers_to_show = list(range(len(decomp)))

# --- Panel 1: Stacked bar — metric vs topological fraction per layer ---
ax = axes[0]
metric_fracs = [d['metric_frac'] for d in decomp]
topo_fracs = [d['topo_frac'] for d in decomp]
x_pos = range(len(decomp))
ax.bar(x_pos, metric_fracs, label='Metric (linear)', color='#2196F3', alpha=0.85)
ax.bar(x_pos, topo_fracs, bottom=metric_fracs, label='Topological (nonlinear)', color='#F44336', alpha=0.85)
ax.set_xlabel('Layer')
ax.set_ylabel('Fraction of ||error||²')
ax.set_title('Error type: metric vs topological')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'L{d["layer"]}' for d in decomp], rotation=45)
ax.legend(loc='lower right')
ax.set_ylim(0, 1.05)

# --- Panel 2: Error magnitude with decomposition ---
ax = axes[1]
norms_metric = [d['norm_metric_mean'] for d in decomp]
norms_topo = [d['norm_topo_mean'] for d in decomp]
ax.bar(x_pos, norms_metric, label='Metric', color='#2196F3', alpha=0.85)
ax.bar(x_pos, norms_topo, bottom=norms_metric, label='Topological', color='#F44336', alpha=0.85)
ax.set_xlabel('Layer')
ax.set_ylabel('Mean ||error|| (activation space)')
ax.set_title('Error magnitude decomposition')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'L{d["layer"]}' for d in decomp], rotation=45)
ax.legend()

# --- Panel 3: SVD rank of metric error ---
ax = axes[2]
r90 = [d['metric_rank_90'] for d in decomp]
r95 = [d['metric_rank_95'] for d in decomp]
r99 = [d['metric_rank_99'] for d in decomp]
ax.plot(x_pos, r90, 'o-', label='rank 90%', color='#4CAF50')
ax.plot(x_pos, r95, 's-', label='rank 95%', color='#FF9800')
ax.plot(x_pos, r99, '^-', label='rank 99%', color='#9C27B0')
ax.axhline(y=HIDDEN_DIM, color='gray', linestyle='--', alpha=0.5, label=f'full rank ({HIDDEN_DIM})')
ax.set_xlabel('Layer')
ax.set_ylabel('Rank needed')
ax.set_title('SVD rank of metric error')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'L{d["layer"]}' for d in decomp], rotation=45)
ax.legend()

plt.tight_layout()
plt.savefig('plots/geometric_decomposition_overview.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Spectral structure of the metric error
#
# The SVD of the metric error matrix at each layer reveals how many
# directions the linearly correctable perturbation actually uses.
# If the singular values decay fast, a low-rank correction suffices.

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()

# Pick 6 layers spread across the network
layer_indices = [0, 2, 4, 7, 9, 11] if len(decomp) >= 12 else list(range(min(6, len(decomp))))

for ax_idx, li in enumerate(layer_indices):
    if ax_idx >= len(axes):
        break
    d = decomp[li]
    ax = axes[ax_idx]
    S = d['metric_svd_S']
    if S.numel() == 0:
        continue
    cumvar = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
    ax.bar(range(len(S)), (S ** 2 / (S ** 2).sum()).numpy(), color='#2196F3', alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(range(len(S)), cumvar.numpy(), 'r-', linewidth=2)
    ax2.axhline(y=0.90, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('Cumulative variance')
    ax.set_title(f'L{li}: rank₉₀={d["metric_rank_90"]}, rank₉₅={d["metric_rank_95"]}')
    ax.set_xlabel('Component')
    ax.set_ylabel('Variance fraction')

plt.suptitle('Singular value spectrum of metric error per layer', fontsize=13)
plt.tight_layout()
plt.savefig('plots/metric_error_svd_spectrum.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Topological distortion: where do hyperplane crossings happen?
#
# The ReLU disagreement mask tells us which neurons flipped, but the
# geometric question is: where in *input space* do these crossings
# concentrate? We expect them near the network's decision boundaries,
# where activations are close to zero (near the hyperplane).

# %%
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.ravel()

PLOT_RANGE = 2.5
GRID_N = 80
grid_1d = np.linspace(-PLOT_RANGE, PLOT_RANGE, GRID_N)
xx, yy = np.meshgrid(grid_1d, grid_1d)
grid_t = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)

float_grid_trace = forward_pass(grid_t, weights, biases)
quant_grid_trace = forward_pass(grid_t, weights_q, biases)

for ax_idx, li in enumerate(layer_indices):
    if ax_idx >= len(axes):
        break
    ax = axes[ax_idx]

    z_float = float_grid_trace.pre_acts[li]
    z_quant = quant_grid_trace.pre_acts[li]

    # Count how many neurons disagree at each grid point
    disagree = (z_float > 0) != (z_quant > 0)  # (N_grid, H)
    n_disagree = disagree.sum(dim=1).float().numpy().reshape(GRID_N, GRID_N)

    im = ax.pcolormesh(xx, yy, n_disagree, cmap='Reds', vmin=0)
    ax.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1],
               c='#1f77b4', s=6, alpha=0.3, edgecolors='none')
    ax.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1],
               c='#d62728', s=6, alpha=0.3, edgecolors='none')
    plt.colorbar(im, ax=ax, label='# neurons crossing')
    ax.set_title(f'L{li}: hyperplane crossings')
    ax.set_xlim(-PLOT_RANGE, PLOT_RANGE)
    ax.set_ylim(-PLOT_RANGE, PLOT_RANGE)
    ax.set_aspect('equal')

plt.suptitle('Topological distortion: where hyperplanes shift enough to cross inputs', fontsize=13)
plt.tight_layout()
plt.savefig('plots/topological_distortion_spatial.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Information loss from topological distortion
#
# Not all hyperplane crossings are equally costly. A neuron that was barely
# active (carrying little variance of the data manifold) costs little when
# it flips. A neuron carrying high-variance structure is expensive to lose.
#
# We compare the variance carried by flipped neurons vs all neurons.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: var_ratio across layers
ax = axes[0]
var_ratios = [d['var_ratio'] for d in decomp]
n_disagree = [d['n_neurons_disagree_ever'] for d in decomp]
ax.bar(range(len(decomp)), var_ratios, color='#F44336', alpha=0.8)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='equal variance')
ax.set_xlabel('Layer')
ax.set_ylabel('Var(flipped) / Var(all)')
ax.set_title('Information content of flipped neurons')
ax.set_xticks(range(len(decomp)))
ax.set_xticklabels([f'L{d["layer"]}' for d in decomp], rotation=45)
ax.legend()

# Panel 2: Number of neurons that ever disagree
ax = axes[1]
ax.bar(range(len(decomp)), n_disagree, color='#9C27B0', alpha=0.8)
ax.axhline(y=HIDDEN_DIM, color='gray', linestyle='--', alpha=0.5, label=f'total ({HIDDEN_DIM})')
ax.set_xlabel('Layer')
ax.set_ylabel('# neurons (ever disagree)')
ax.set_title('How many neurons ever cross a hyperplane')
ax.set_xticks(range(len(decomp)))
ax.set_xticklabels([f'L{d["layer"]}' for d in decomp], rotation=45)
ax.legend()

plt.tight_layout()
plt.savefig('plots/topological_information_loss.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Spectral properties of $E_L$ and connection to metric distortion
#
# The weight perturbation $E_L = W_q - W$ is the geometric distortion of
# the linear transform. Its spectral norm $||E_L||_2$ is the worst-case
# directional amplification of the perturbation; the ratio
# $||E_L||_2 / ||E_L||_F$ tells us how concentrated the distortion is along
# a few directions (high ratio = concentrated = low-rank correction suffices).

# %%
print(f"{'Layer':>5}  {'||E||_2':>8}  {'||E||_F':>8}  "
      f"{'ratio':>6}  {'metric%':>8}  {'rank95':>6}")
print("-" * 55)
for d in decomp:
    ratio = d['E_spectral_norm'] / d['E_frobenius_norm'] if d['E_frobenius_norm'] > 0 else 0
    print(f"  L{d['layer']:<3}  {d['E_spectral_norm']:8.4f}  {d['E_frobenius_norm']:8.4f}  "
          f"{ratio:6.3f}  {d['metric_frac']:7.1%}  {d['metric_rank_95']:6d}")

# %% [markdown]
# ## Rank-$k$ recovery of metric error
#
# If the metric error is low-rank, we can approximate correction with a
# rank-$k$ matrix and measure how much of the total error we recover.
# This directly predicts the performance of a low-rank learned correction.

# %%
@torch.no_grad()
def rank_k_recovery(decomp_results, float_trace, quant_trace):
    """For each layer, measure output error after rank-k correction of metric error.

    Uses SVD of the per-layer metric error to construct rank-k corrections,
    then propagates through remaining layers to measure output impact.
    """
    results = []
    for d in decomp_results:
        i = d['layer']
        eps_metric = d['eps_metric']   # (N, H)
        eps_total = d['eps_total']     # (N, H)

        S = d['metric_svd_S']
        total_var = (S ** 2).sum()
        if total_var == 0:
            results.append({"layer": i, "ranks": [], "recovery": []})
            continue

        # Test a range of ranks
        max_rank = min(S.shape[0], HIDDEN_DIM)
        test_ranks = sorted(set([1, 2, 3, 5, 8, 10, 16, max_rank]))
        test_ranks = [r for r in test_ranks if r <= max_rank]

        # Total error energy (baseline: no correction)
        total_energy = (eps_total ** 2).sum().item()
        # Pure metric energy
        metric_energy = (eps_metric ** 2).sum().item()

        recoveries = []
        for k in test_ranks:
            # Rank-k approximation of metric error captures this much
            captured = (S[:k] ** 2).sum().item() / total_var.item()
            # Fraction of total error that rank-k metric correction removes
            recovery = captured * metric_energy / total_energy if total_energy > 0 else 0
            recoveries.append(recovery)

        results.append({
            "layer": i,
            "ranks": test_ranks,
            "recovery": recoveries,
            "metric_frac": d['metric_frac'],
        })
    return results


rank_results = rank_k_recovery(decomp, float_trace, quant_trace)

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()

for ax_idx, li in enumerate(layer_indices):
    if ax_idx >= len(axes):
        break
    ax = axes[ax_idx]
    rr = rank_results[li]
    if not rr['ranks']:
        continue
    ax.plot(rr['ranks'], rr['recovery'], 'o-', color='#2196F3', linewidth=2)
    ax.axhline(y=rr['metric_frac'], color='#F44336', linestyle='--',
               label=f'metric ceiling ({rr["metric_frac"]:.1%})')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Rank k')
    ax.set_ylabel('Fraction of ||error||² recovered')
    ax.set_title(f'L{li}')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

plt.suptitle('Rank-k correction recovery per layer', fontsize=13)
plt.tight_layout()
plt.savefig('plots/rank_k_recovery.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Metric-predicted rank vs SVD ground truth
#
# Can the spectral ratio $||E_L||_2 / ||E_L||_F$ predict the SVD rank
# needed for 95% recovery without computing the full SVD on calibration
# data? If so, we have a metric-driven design rule.

# %%
fig, ax = plt.subplots(figsize=(7, 5))

spectral_ratios = [d['E_spectral_norm'] / d['E_frobenius_norm']
                   if d['E_frobenius_norm'] > 0 else 0 for d in decomp]
ranks_95 = [d['metric_rank_95'] for d in decomp]

ax.scatter(spectral_ratios, ranks_95, c=[d['layer'] for d in decomp],
           cmap='viridis', s=80, zorder=3)
for d in decomp:
    ratio = d['E_spectral_norm'] / d['E_frobenius_norm'] if d['E_frobenius_norm'] > 0 else 0
    ax.annotate(f'L{d["layer"]}', (ratio, d['metric_rank_95']),
                textcoords='offset points', xytext=(5, 5), fontsize=8)

ax.set_xlabel('||E_L||₂ / ||E_L||_F (spectral concentration)')
ax.set_ylabel('SVD rank for 95% metric recovery')
ax.set_title('Spectral ratio as rank predictor')
plt.tight_layout()
plt.savefig('plots/spectral_ratio_vs_rank.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Summary
#
# This decomposition gives us the per-layer budget for correction design:
#
# 1. **Metric fraction** = ceiling for any linear/low-rank correction.
#    If 95% of error is metric, even a perfect nonlinear corrector can only
#    improve on linear by 5%.
#
# 2. **SVD rank** of the metric error = how many parameters the linear
#    correction needs. Low rank → very cheap correction.
#
# 3. **Topological fraction** × **var_ratio** = actual information loss.
#    If flipped neurons carried little variance, the topological error is
#    cosmetic. If they carried high variance, it's a real loss.
#
# 4. **Spectral ratio** of $E_L$ predicts the correction rank without
#    needing calibration data SVD.
#
# These measurements directly inform Experiment 1 (below) and Experiment 2
# (global error trajectory correction, future work).

# %% [markdown]
# ---
# # Experiment 1: Metric-Guided Low-Rank Correction
#
# We know from Experiment 3 that the correction problem is effectively linear
# and low-rank. Now we build actual rank-$k$ corrections and validate:
#
# 1. **Task-level recovery**: does rank-$k$ correction at each layer recover
#    float accuracy? (Not just error energy — actual classification performance.)
#
# 2. **Metric-guided rank selection**: can geometric metrics predict the rank
#    needed at each layer without computing the calibration-set SVD?
#
# ### Approach
#
# At each hidden layer $L$, the oracle correction on a calibration set
# produces a matrix $C_L \in \mathbb{R}^{N \times H}$. The SVD of $C_L$
# reveals the correction subspace. A rank-$k$ correction projects the oracle
# correction onto its top-$k$ singular directions — this is the best possible
# rank-$k$ linear correction (by Eckart–Young).
#
# We apply these per-layer rank-$k$ corrections during a forward pass and
# measure task accuracy. Then we check which geometric metrics predict the
# needed rank.

# %%
@torch.no_grad()
def compute_oracle_corrections(x, weights, biases, weights_q, float_trace):
    """Compute the oracle correction vector at each hidden layer.

    C_L = -E_L @ â_{L-1} - W_L @ ε_{L-1}

    Returns list of (N, H) correction tensors, one per hidden layer.
    """
    a = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    epsilon = torch.zeros_like(a)
    corrections = []

    for i in range(len(weights)):
        is_last = (i == len(weights) - 1)
        E = weights_q[i] - weights[i]
        W, Wq, b = weights[i], weights_q[i], biases[i]

        C = -F.linear(a, E) - F.linear(epsilon, W)
        z = F.linear(a, Wq, b) + C
        a_new = z if is_last else F.relu(z)
        epsilon = a_new - float_trace.post_acts[i]

        if not is_last:
            corrections.append(C)
        a = a_new

    return corrections


@torch.no_grad()
def rank_k_correction_bases(corrections):
    """Compute SVD of oracle corrections at each layer.

    Returns list of (U, S, Vh) per layer, where rank-k correction is
    reconstructed as U[:, :k] @ diag(S[:k]) @ Vh[:k, :].
    """
    bases = []
    for C in corrections:
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        bases.append((U, S, Vh))
    return bases


@torch.no_grad()
def forward_with_rank_k_correction(x, weights, biases, weights_q,
                                    bases, ranks):
    """Forward pass applying rank-k correction at each hidden layer.

    Args:
        bases: list of (U, S, Vh) from rank_k_correction_bases
        ranks: list of int, rank to use at each hidden layer
    """
    a = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

    for i, (Wq, b) in enumerate(zip(weights_q, biases)):
        z = F.linear(a, Wq, b)
        is_last = (i == len(weights_q) - 1)
        if not is_last:
            # Apply rank-k correction to pre-activation (before ReLU)
            U, S, Vh = bases[i]
            k = ranks[i]
            if k > 0 and k <= len(S):
                C_k = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]
                z = z + C_k
        a = z if is_last else F.relu(z)
    return a


# %%
# Compute oracle corrections and their SVD bases
oracle_corrections = compute_oracle_corrections(X_t, weights, biases, weights_q,
                                                 float_trace)
svd_bases = rank_k_correction_bases(oracle_corrections)

print("Oracle correction SVD — singular value decay per layer:")
for i, (U, S, Vh) in enumerate(svd_bases):
    cumvar = torch.cumsum(S**2, dim=0) / (S**2).sum()
    r90 = int((cumvar < 0.90).sum().item()) + 1
    r95 = int((cumvar < 0.95).sum().item()) + 1
    r99 = int((cumvar < 0.99).sum().item()) + 1
    print(f"  L{i}: rank₉₀={r90}, rank₉₅={r95}, rank₉₉={r99}  "
          f"(top-1 captures {cumvar[0].item():.1%})")

# %% [markdown]
# ### Rank-$k$ correction → task accuracy
#
# For a range of uniform ranks (same $k$ at every layer), measure the
# classification accuracy of the corrected model. This shows the
# accuracy-vs-parameters tradeoff.

# %%
y_t = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)

# Baseline accuracies
acc_float = ((torch.sigmoid(float_trace.post_acts[-1]) > 0.5).float() == y_t).float().mean().item()
acc_quant = ((torch.sigmoid(quant_trace.post_acts[-1]) > 0.5).float() == y_t).float().mean().item()
print(f"Float accuracy:     {acc_float:.1%}")
print(f"Quantized accuracy: {acc_quant:.1%}")

n_hidden = len(svd_bases)
test_ranks = [0, 1, 2, 3, 5, 8, 10, 16, HIDDEN_DIM]
test_ranks = [r for r in test_ranks if r <= HIDDEN_DIM]

rank_accuracies = []
for k in test_ranks:
    ranks = [k] * n_hidden
    logits = forward_with_rank_k_correction(X_t, weights, biases, weights_q,
                                             svd_bases, ranks)
    acc_k = ((torch.sigmoid(logits) > 0.5).float() == y_t).float().mean().item()
    rank_accuracies.append(acc_k)
    print(f"  Rank {k:>2d}: {acc_k:.1%}")

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(test_ranks, rank_accuracies, 'o-', color='#2196F3', linewidth=2, markersize=8)
ax.axhline(y=acc_float, color='green', linestyle='--', label=f'Float ({acc_float:.1%})')
ax.axhline(y=acc_quant, color='red', linestyle='--', label=f'Quantized ({acc_quant:.1%})')
ax.set_xlabel('Correction rank (uniform across layers)')
ax.set_ylabel('Accuracy')
ax.set_title('Task accuracy vs correction rank')
ax.legend()
ax.set_ylim(min(acc_quant - 0.05, 0.45), 1.0)
plt.tight_layout()
plt.savefig('plots/rank_k_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Per-layer rank selection via geometric metrics
#
# Instead of a uniform rank, can we use per-layer geometric metrics to
# pick the right rank at each layer? Candidate predictors:
#
# - **Cumulative amplification** $||T_L||_2$: how much the network stretches
#   input-to-layer-$L$. High amplification → error more concentrated → lower
#   rank needed.
# - **Layer depth** (normalized $L / D$): a proxy for cumulative amplification.
# - **Spectral concentration** $||E_L||_2 / ||E_L||_F$: how concentrated the
#   weight perturbation is along a few directions.
# - **Weight spectral norm** $||W_L||_2$: how much this layer amplifies.

# %%
# Collect per-layer metrics and ground-truth ranks
layer_metrics = []
for i in range(n_hidden):
    d = decomp[i]
    E = weights_q[i] - weights[i]
    W = weights[i]

    # Ground-truth rank from oracle correction SVD
    U, S, Vh = svd_bases[i]
    cumvar = torch.cumsum(S**2, dim=0) / (S**2).sum()
    gt_rank_95 = int((cumvar < 0.95).sum().item()) + 1

    layer_metrics.append({
        "layer": i,
        "depth_frac": i / n_hidden,
        "gt_rank_95": gt_rank_95,
        "cum_amplification": tracker.cumulative_amplification(i),
        "E_spectral_ratio": d['E_spectral_norm'] / d['E_frobenius_norm']
            if d['E_frobenius_norm'] > 0 else 0,
        "W_spectral_norm": torch.linalg.norm(W, ord=2).item(),
        "metric_rank_95": d['metric_rank_95'],  # from Experiment 3
    })

print(f"{'Layer':>5}  {'Depth%':>6}  {'||T||₂':>8}  {'E_ratio':>7}  "
      f"{'||W||₂':>7}  {'GTrank':>6}  {'MetRank':>7}")
print("-" * 60)
for m in layer_metrics:
    print(f"  L{m['layer']:<3}  {m['depth_frac']:5.0%}  {m['cum_amplification']:8.1f}  "
          f"{m['E_spectral_ratio']:7.3f}  {m['W_spectral_norm']:7.3f}  "
          f"{m['gt_rank_95']:6d}  {m['metric_rank_95']:7d}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

gt_ranks = [m['gt_rank_95'] for m in layer_metrics]

# Panel 1: depth vs rank
ax = axes[0]
depths = [m['depth_frac'] for m in layer_metrics]
ax.scatter(depths, gt_ranks, c='#2196F3', s=80, zorder=3)
for m in layer_metrics:
    ax.annotate(f'L{m["layer"]}', (m['depth_frac'], m['gt_rank_95']),
                textcoords='offset points', xytext=(5, 5), fontsize=8)
ax.set_xlabel('Relative depth (L/D)')
ax.set_ylabel('SVD rank for 95% correction')
ax.set_title('Depth as rank predictor')

# Panel 2: cumulative amplification vs rank
ax = axes[1]
amps = [m['cum_amplification'] for m in layer_metrics]
ax.scatter(amps, gt_ranks, c='#4CAF50', s=80, zorder=3)
for m in layer_metrics:
    ax.annotate(f'L{m["layer"]}', (m['cum_amplification'], m['gt_rank_95']),
                textcoords='offset points', xytext=(5, 5), fontsize=8)
ax.set_xlabel('||T_L||₂ (cumulative amplification)')
ax.set_ylabel('SVD rank for 95% correction')
ax.set_title('Cumulative amplification as rank predictor')
ax.set_xscale('log')

# Panel 3: metric error rank vs oracle correction rank
ax = axes[2]
met_ranks = [m['metric_rank_95'] for m in layer_metrics]
ax.scatter(met_ranks, gt_ranks, c='#FF9800', s=80, zorder=3)
for m in layer_metrics:
    ax.annotate(f'L{m["layer"]}', (m['metric_rank_95'], m['gt_rank_95']),
                textcoords='offset points', xytext=(5, 5), fontsize=8)
# Diagonal reference
max_r = max(max(met_ranks), max(gt_ranks))
ax.plot([0, max_r], [0, max_r], 'k--', alpha=0.3)
ax.set_xlabel('Metric error rank₉₅ (Experiment 3)')
ax.set_ylabel('Oracle correction rank₉₅')
ax.set_title('Metric error rank as correction rank predictor')

plt.tight_layout()
plt.savefig('plots/rank_predictors.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Predicted-rank correction
#
# Use the best predictor to assign per-layer ranks, then measure accuracy.
# Compare against uniform-rank baselines and the oracle (full-rank).

# %%
# Strategy 1: use metric error rank from Experiment 3 as the predicted rank
predicted_ranks = [m['metric_rank_95'] for m in layer_metrics]

# Strategy 2: simple depth heuristic — later layers need less rank
# Fit: rank = max_rank * (1 - depth_frac) + 1
max_metric_rank = max(m['metric_rank_95'] for m in layer_metrics)
depth_ranks = [max(1, round(max_metric_rank * (1 - m['depth_frac']) + 0.5))
               for m in layer_metrics]

strategies = [
    ("No correction",       [0] * n_hidden),
    ("Uniform rank-1",      [1] * n_hidden),
    ("Uniform rank-2",      [2] * n_hidden),
    ("Uniform rank-3",      [3] * n_hidden),
    ("Uniform rank-5",      [5] * n_hidden),
    ("Metric-predicted",    predicted_ranks),
    ("Depth heuristic",     depth_ranks),
    ("Full rank (oracle)",  [HIDDEN_DIM] * n_hidden),
]

total_params_per_rank = lambda ranks: sum(
    r * (weights[i].shape[1] + weights[i].shape[0]) for i, r in enumerate(ranks)
    if i < n_hidden
)

print(f"{'Strategy':<22s}  {'Accuracy':>8s}  {'Params':>7s}  {'Ranks':>30s}")
print("-" * 75)
for name, ranks in strategies:
    logits = forward_with_rank_k_correction(X_t, weights, biases, weights_q,
                                             svd_bases, ranks)
    acc_s = ((torch.sigmoid(logits) > 0.5).float() == y_t).float().mean().item()
    n_params = total_params_per_rank(ranks)
    rank_str = str(ranks[:4]) + "..." if len(ranks) > 4 else str(ranks)
    print(f"{name:<22s}  {acc_s:7.1%}  {n_params:7d}  {rank_str:>30s}")

# %% [markdown]
# ### Per-layer rank: predicted vs ground truth
#
# How well does the metric error rank (from Experiment 3, no oracle needed
# at deployment) match the oracle correction rank?

# %%
fig, ax = plt.subplots(figsize=(7, 5))

x_pos = range(n_hidden)
ax.bar([x - 0.15 for x in x_pos], gt_ranks, width=0.3,
       label='Oracle correction rank₉₅', color='#2196F3', alpha=0.8)
ax.bar([x + 0.15 for x in x_pos], predicted_ranks, width=0.3,
       label='Metric error rank₉₅ (predicted)', color='#FF9800', alpha=0.8)
ax.set_xlabel('Layer')
ax.set_ylabel('Rank')
ax.set_title('Predicted vs ground-truth correction rank')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'L{i}' for i in range(n_hidden)])
ax.legend()
plt.tight_layout()
plt.savefig('plots/predicted_vs_gt_rank.png', dpi=150, bbox_inches='tight')
plt.show()
