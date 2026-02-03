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
# # Classification Quantization Geometry
#
# How does weight quantization affect a trained binary classifier? This notebook
# trains small MLPs (2→8→8→1) on 2D datasets and visualizes quantization error
# propagation through layers — extending the 2D geometric analysis to higher
# dimensions using PCA projections.
#
# ## Key Findings
#
# 1. **8-bit quantization is nearly invisible**: relative error is ~0.1% per layer,
#    zero accuracy degradation, decision boundaries are visually identical.
# 2. **PCA captures 90%+ variance in 2 components**: the 8D activation space is
#    low-rank enough that 2D projections are faithful.
# 3. **Class separation improves through layers**: PCA projections show the network
#    progressively disentangling the classes.
# 4. **Error concentrates near decision boundaries**: points close to the boundary
#    have higher relative error because their activations are small.
# 5. **Flip rate is near zero at 8-bit**: with 8 neurons, the continuous survival/flip/dead
#    rates replace the binary 2D categories. Almost all components either survive or die
#    identically in both paths.
# 6. **Dead rate is high (~40-75%)**: most neurons are inactive for any given input,
#    consistent with ReLU sparsity. Dead neurons contribute zero error.
#
# ## Structure
# - **Cell 1**: Setup, dataset generation, model training, weight extraction
# - **Cell 2**: Analysis functions (propagation, PCA visualization, decision boundaries)
# - **Cell 3**: Run analysis and generate plots

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA

from aleph.qgeom import quantize
from aleph.datasets import make_spirals

# Configuration for this notebook
BITS = 8
DELTA = 1.0 / (2 ** (BITS - 1))

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# MODEL + TRAINING
# ============================================================

def make_mlp(hidden_dim=8, depth=2, input_dim=2, output_dim=1):
    """Configurable MLP with ReLU activations.

    Args:
        hidden_dim: Number of neurons in each hidden layer
        depth: Number of hidden layers
        input_dim: Input dimension
        output_dim: Output dimension

    Returns:
        nn.Sequential model: input_dim -> (hidden_dim -> ReLU) * depth -> output_dim
    """
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def train_model(X, y, epochs=1000, lr=0.01):
    """Train binary classifier with BCE loss."""
    model = make_mlp()
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

        if (epoch + 1) % 200 == 0:
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y_t).float().mean()
            print(f'  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc.item():.1%}')

    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y_t).float().mean().item()
    return model, acc


def extract_weights(model):
    """Extract weight matrices and biases as numpy arrays.

    Returns (float_weights, biases, quant_weights).
    Only includes Linear layers (skips ReLU).
    """
    float_weights = []
    biases = []
    for module in model:
        if isinstance(module, nn.Linear):
            W = module.weight.detach().numpy().copy()
            b = module.bias.detach().numpy().copy()
            float_weights.append(W)
            biases.append(b)
    quant_weights = [quantize(W, DELTA) for W in float_weights]
    return float_weights, biases, quant_weights


def full_forward(points, weights, biases):
    """Full forward pass through MLP, return final logits."""
    val = points.copy()
    for i, (W, b) in enumerate(zip(weights, biases)):
        val = val @ W.T + b
        if i < len(weights) - 1:
            val = np.maximum(0, val)
    return val


# ============================================================
# Generate datasets and train
# ============================================================

datasets = {}
X_moons, y_moons = make_moons(n_samples=500, noise=0.15, random_state=42)
datasets['moons'] = (X_moons, y_moons)

X_spirals, y_spirals = make_spirals(n_samples=500, noise=0.3, random_state=42)
datasets['spirals'] = (X_spirals, y_spirals)

trained = {}
for name, (X, y) in datasets.items():
    print(f'\nTraining on {name}...')
    model, acc = train_model(X, y)
    float_w, bs, quant_w = extract_weights(model)
    logits_q = full_forward(X, quant_w, bs).ravel()
    quant_acc = np.mean(((1 / (1 + np.exp(-logits_q))) > 0.5) == y)
    trained[name] = {
        'X': X, 'y': y,
        'float_weights': float_w, 'biases': bs, 'quant_weights': quant_w,
        'float_acc': acc, 'quant_acc': quant_acc,
    }
    print(f'  Float accuracy: {acc:.1%}')
    print(f'  Quant accuracy: {quant_acc:.1%}')

print(f'\nFramework loaded. BITS={BITS}, DELTA={DELTA:.6f}')

# %%
# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================


def forward_pass(points, weights, biases):
    """Numpy forward pass. Returns list of dicts with pre/post per layer."""
    val = points.copy()
    layers = []
    n_layers = len(weights)

    for i, (W, b) in enumerate(zip(weights, biases)):
        pre = val @ W.T + b
        if i < n_layers - 1:
            post = np.maximum(0, pre)  # ReLU
        else:
            post = pre  # final layer, no ReLU
        layers.append({'pre': pre, 'post': post})
        val = post

    return layers


def propagate_classification(points, float_weights, biases, quant_weights):
    """Propagate through float and quant paths, compute fate stats.

    Returns per-layer dicts with activations, error, and continuous
    component-level metrics (survival_rate, flip_rate, dead_rate).
    """
    float_layers = forward_pass(points, float_weights, biases)
    quant_layers = forward_pass(points, quant_weights, biases)
    n_layers = len(float_weights)

    results = []
    for i in range(n_layers):
        fl = float_layers[i]
        ql = quant_layers[i]
        error = ql['post'] - fl['post']
        is_hidden = i < n_layers - 1

        layer_dict = {
            'float_pre': fl['pre'],
            'float_post': fl['post'],
            'quant_pre': ql['pre'],
            'quant_post': ql['post'],
            'error': error,
            'is_hidden': is_hidden,
        }

        if is_hidden:
            float_pos = fl['pre'] >= 0
            quant_pos = ql['pre'] >= 0
            comp_surviving = float_pos & quant_pos
            comp_dead = ~float_pos & ~quant_pos
            comp_flipped = float_pos != quant_pos

            width = fl['pre'].shape[1]
            survival_rate = comp_surviving.sum(axis=1) / width
            flip_rate = comp_flipped.sum(axis=1) / width
            dead_rate = comp_dead.sum(axis=1) / width

            float_norm = np.linalg.norm(fl['post'], axis=1)
            error_norm = np.linalg.norm(error, axis=1)
            rel_error = np.where(float_norm > 1e-10,
                                 error_norm / float_norm,
                                 error_norm)

            layer_dict.update({
                'comp_surviving': comp_surviving,
                'comp_dead': comp_dead,
                'comp_flipped': comp_flipped,
                'survival_rate': survival_rate,
                'flip_rate': flip_rate,
                'dead_rate': dead_rate,
                'rel_error': rel_error,
                'stats': {
                    'mean_survival_rate': survival_rate.mean(),
                    'mean_flip_rate': flip_rate.mean(),
                    'mean_dead_rate': dead_rate.mean(),
                    'mean_rel_error': rel_error.mean(),
                    'max_rel_error': rel_error.max(),
                    'n_any_flip': int((flip_rate > 0).sum()),
                    'frac_any_flip': (flip_rate > 0).mean(),
                },
            })
        else:
            abs_error = np.abs(error.ravel())
            layer_dict['stats'] = {
                'mean_abs_error': abs_error.mean(),
                'max_abs_error': abs_error.max(),
            }

        results.append(layer_dict)

    return results


# ============================================================
# VISUALIZATION
# ============================================================


def _square_lims(ax, margin=1.05):
    """Equalize axis ranges to make subplot square."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cx, cy = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
    half = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2 * margin
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def plot_decision_boundary(ax, X, y, float_w, biases, quant_w, extent=None):
    """Draw decision boundaries for float and quant networks."""
    if extent is None:
        margin = 0.5
        x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
        y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    else:
        x_min, x_max, y_min, y_max = extent

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                          np.linspace(y_min, y_max, 200))
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    logits_f = full_forward(grid, float_w, biases).ravel()
    probs_f = 1 / (1 + np.exp(-logits_f))
    Z_f = probs_f.reshape(xx.shape)

    logits_q = full_forward(grid, quant_w, biases).ravel()
    probs_q = 1 / (1 + np.exp(-logits_q))
    Z_q = probs_q.reshape(xx.shape)

    ax.contourf(xx, yy, (Z_f > 0.5).astype(float), levels=[0, 0.5, 1],
                colors=['#aec7e8', '#ffbdbd'], alpha=0.4)
    ax.contour(xx, yy, Z_f, levels=[0.5], colors=['#1f77b4'], linewidths=2,
               linestyles='-')
    ax.contour(xx, yy, Z_q, levels=[0.5], colors=['#d62728'], linewidths=2,
               linestyles='--')

    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='#1f77b4', s=12, edgecolors='k',
               linewidth=0.3, zorder=5)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='#d62728', s=12, edgecolors='k',
               linewidth=0.3, zorder=5)


def plot_classification_evolution(X, y, float_w, biases, quant_w, dataset_name):
    """Main grid: data row (PCA of activations) + error row (input space).

    Data row: PCA-2D projection of activations at each layer.
      Filled = float, hollow = quant, color = class.
    Error row: input space colored by error metrics.
      Col 0 = decision boundary, cols 1+ = relative error heatmap.
    """
    layer_data = propagate_classification(X, float_w, biases, quant_w)

    n_cols = len(layer_data) + 1
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    margin = 0.5
    extent = [X[:, 0].min() - margin, X[:, 0].max() + margin,
              X[:, 1].min() - margin, X[:, 1].max() + margin]

    # === DATA ROW ===
    ax = axes[0, 0]
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='#1f77b4', s=15, edgecolors='k',
               linewidth=0.3, label='class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='#d62728', s=15, edgecolors='k',
               linewidth=0.3, label='class 1')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    ax.set_title('Input (2D)', fontweight='bold')
    ax.set_ylabel('data', fontsize=11, fontweight='bold')
    _square_lims(ax)

    for col, ld in enumerate(layer_data):
        ax = axes[0, col + 1]
        float_act = ld['float_post']
        quant_act = ld['quant_post']

        if float_act.shape[1] > 2:
            pca = PCA(n_components=2)
            float_2d = pca.fit_transform(float_act)
            quant_2d = pca.transform(quant_act)
            dim_label = 'PCA %d%%' % int(pca.explained_variance_ratio_.sum() * 100)
        elif float_act.shape[1] == 2:
            float_2d = float_act
            quant_2d = quant_act
            dim_label = '2D'
        else:
            float_2d = np.column_stack([float_act.ravel(), y * 0.1])
            quant_2d = np.column_stack([quant_act.ravel(), y * 0.1])
            dim_label = '1D logit'

        # Float: filled
        ax.scatter(float_2d[y == 0, 0], float_2d[y == 0, 1], c='#1f77b4',
                   s=15, edgecolors='none', alpha=0.7)
        ax.scatter(float_2d[y == 1, 0], float_2d[y == 1, 1], c='#d62728',
                   s=15, edgecolors='none', alpha=0.7)

        # Quant: hollow
        ax.scatter(quant_2d[y == 0, 0], quant_2d[y == 0, 1], facecolors='none',
                   edgecolors='#1f77b4', s=20, linewidth=0.5, alpha=0.5)
        ax.scatter(quant_2d[y == 1, 0], quant_2d[y == 1, 1], facecolors='none',
                   edgecolors='#d62728', s=20, linewidth=0.5, alpha=0.5)

        layer_label = 'L%d' % (col + 1) if ld['is_hidden'] else 'Output'
        width = float_act.shape[1]
        ax.set_title('%s (%dD -> %s)' % (layer_label, width, dim_label),
                     fontweight='bold')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        _square_lims(ax)

        if ld['is_hidden']:
            s = ld['stats']
            txt = ('surv: %.0f%%\nflip: %.0f%%\ndead: %.0f%%'
                   % (s['mean_survival_rate'] * 100,
                      s['mean_flip_rate'] * 100,
                      s['mean_dead_rate'] * 100))
            ax.text(0.95, 0.95, txt,
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=7, family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.8))

    # === ERROR ROW ===
    ax = axes[1, 0]
    plot_decision_boundary(ax, X, y, float_w, biases, quant_w, extent)
    ax.set_title('Decision boundary\nsolid=float, dashed=quant', fontweight='bold')
    ax.set_ylabel('error', fontsize=11, fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)

    for col, ld in enumerate(layer_data):
        ax = axes[1, col + 1]

        if ld['is_hidden']:
            rel_err = ld['rel_error']
            scatter = ax.scatter(X[:, 0], X[:, 1], c=rel_err, cmap='hot_r',
                                 s=15, edgecolors='none', vmin=0)
            plt.colorbar(scatter, ax=ax, label='rel. error', shrink=0.8)

            has_flip = ld['flip_rate'] > 0
            if has_flip.any():
                ax.scatter(X[has_flip, 0], X[has_flip, 1],
                           facecolors='none', edgecolors='lime', s=40,
                           linewidth=1.5, zorder=5,
                           label='flips (%d)' % has_flip.sum())
                ax.legend(fontsize=6, loc='lower right')

            s = ld['stats']
            ax.set_title('L%d rel. error (mean=%.4f)'
                         % (col + 1, s['mean_rel_error']),
                         fontweight='bold')
        else:
            abs_err = np.abs(ld['error'].ravel())
            scatter = ax.scatter(X[:, 0], X[:, 1], c=abs_err, cmap='hot_r',
                                 s=15, edgecolors='none', vmin=0)
            plt.colorbar(scatter, ax=ax, label='|logit error|', shrink=0.8)
            s = ld['stats']
            ax.set_title('Output |logit error| (mean=%.4f)'
                         % s['mean_abs_error'],
                         fontweight='bold')

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)

    plt.suptitle('%s: Weight Quantization Effect on Classification\n'
                 'Data row: activations (filled=float, hollow=quant)  |  '
                 'Error row: quantization error in input space'
                 % dataset_name.upper(),
                 fontsize=12, y=1.04)
    plt.tight_layout()
    plt.savefig('plots/classification_%s.png' % dataset_name,
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_error_summary(all_trained):
    """Summary: error metrics across layers for all datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    dataset_colors = {'moons': '#1f77b4', 'spirals': '#ff7f0e'}

    for name, data in all_trained.items():
        layer_data = propagate_classification(
            data['X'], data['float_weights'], data['biases'], data['quant_weights'])
        color = dataset_colors.get(name, 'gray')

        hidden_layers = [(i, ld) for i, ld in enumerate(layer_data) if ld['is_hidden']]
        layer_nums = [i + 1 for i, _ in hidden_layers]

        # Relative error
        ax = axes[0]
        mean_errs = [ld['stats']['mean_rel_error'] for _, ld in hidden_layers]
        max_errs = [ld['stats']['max_rel_error'] for _, ld in hidden_layers]
        ax.plot(layer_nums, mean_errs, 'o-', color=color, linewidth=2,
                markersize=6, label='%s (mean)' % name)
        ax.plot(layer_nums, max_errs, 's--', color=color, linewidth=1,
                markersize=5, alpha=0.5, label='%s (max)' % name)

        # Flip rate
        ax = axes[1]
        flip_rates = [ld['stats']['mean_flip_rate'] for _, ld in hidden_layers]
        ax.plot(layer_nums, flip_rates, 'o-', color=color, linewidth=2,
                markersize=6, label=name)

        # Survival rate
        ax = axes[2]
        surv_rates = [ld['stats']['mean_survival_rate'] for _, ld in hidden_layers]
        ax.plot(layer_nums, surv_rates, 'o-', color=color, linewidth=2,
                markersize=6, label=name)

    for ax, title, ylabel in zip(axes,
            ['Relative Error per Layer', 'Mean Flip Rate per Layer',
             'Mean Survival Rate per Layer'],
            ['Relative error', 'Flip rate', 'Survival rate']):
        ax.set_xlabel('Layer')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['L1', 'L2'])

    plt.tight_layout()
    plt.savefig('plots/classification_error_summary.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# %%
# ============================================================
# RUN ANALYSIS
# ============================================================

print('=' * 70)
print('CLASSIFICATION QUANTIZATION GEOMETRY')
print('=' * 70)

for name, data in trained.items():
    print(f'\n--- {name.upper()} ---')
    print(f'  Float accuracy: {data["float_acc"]:.1%}')
    print(f'  Quant accuracy: {data["quant_acc"]:.1%}')

    layer_data = propagate_classification(
        data['X'], data['float_weights'], data['biases'], data['quant_weights'])

    for i, ld in enumerate(layer_data):
        if ld['is_hidden']:
            s = ld['stats']
            print(f'  L{i+1}: surv={s["mean_survival_rate"]:.1%}, '
                  f'flip={s["mean_flip_rate"]:.1%}, '
                  f'dead={s["mean_dead_rate"]:.1%}, '
                  f'rel_err={s["mean_rel_error"]:.6f} '
                  f'(max={s["max_rel_error"]:.6f})')
        else:
            s = ld['stats']
            print(f'  Output: mean_abs_err={s["mean_abs_error"]:.6f}, '
                  f'max_abs_err={s["max_abs_error"]:.6f}')

    plot_classification_evolution(
        data['X'], data['y'],
        data['float_weights'], data['biases'], data['quant_weights'],
        name)

plot_error_summary(trained)

print('\nOBSERVATIONS:')
print('1. 8-bit weight quantization has zero accuracy impact on these small networks')
print('2. Relative error is ~0.1% per layer, accumulating to ~0.03 logit error at output')
print('3. ReLU flip rate is near zero: quantization error is too small to change ReLU decisions')
print('4. Error concentrates near the decision boundary (small activations = high relative error)')
print('5. PCA captures 90%+ variance, showing the 8D space is low-rank for these datasets')
print('6. Dead rate ~40-75%: most neurons inactive per input (ReLU sparsity), contributing zero error')

# %%
