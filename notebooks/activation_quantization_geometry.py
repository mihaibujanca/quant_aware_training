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
# # Activation Quantization Error Geometry
#
# Extends weight-only analysis to **full fake quantization** (weight + activation),
# simulating real integer inference:
# `int8_input × int8_weight → int32_accumulator (exact) → requantize → int8_output`
#
# Activation quantization is **per-tensor**: one scale factor for the entire activation
# matrix at each layer. This is the standard approach in TFLite, ONNX Runtime, and
# PyTorch's PTQ/QAT pipelines.
#
# ## Key Findings
#
# ### Weight-only vs Full Quantization
# 1. **Weight-only error is smooth**: The error is a fixed linear transform of input,
#    so the error manifold preserves topology (circles stay ellipses, lines stay lines).
# 2. **Activation quantization adds discretization artifacts**: Per-tensor rounding
#    creates grid-aligned jumps. Error manifolds become stepped — smooth curves develop
#    jagged edges from rounding at grid boundaries.
# 3. **Activation error can dominate**: For near-identity weights, weight quantization
#    error is tiny (~0.003) while activation rounding error scales with signal amplitude
#    (~0.08 at int8 for activations of magnitude ~20).
#
# ### Bit-width Comparison
# 4. **8-bit**: Error manifolds are nearly smooth. Activation quantization adds minor
#    perturbation over weight-only error.
# 5. **4-bit**: Visible discretization. Rounding grid is coarse enough that nearby
#    points can collapse to the same grid point.
# 6. **2-bit**: Severe quantization. Only 4 levels per axis — small components get
#    zeroed out entirely. Manifold structure is largely destroyed.
#
# ### Error Growth
# 7. **Superlinear accumulation**: Unlike weight-only error (linear growth via Minkowski
#    sum), full quantization error grows superlinearly because activation rounding
#    compounds with weight rounding at each layer.
#
# ## Structure
# - **Cell 1**: Setup, fake quantization functions, forward pass simulation
# - **Cell 2**: Visualizations (weight-only vs full, error evolution, bit-width comparison)

# %%
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from aleph.qgeom import make_manifold


# ============================================================
# Fake quantization
# ============================================================

def fake_quantize(x, bits=8):
    """Symmetric fake quantization: float32 → int{bits} → float32.

    scale = max(|x|) / (2^(bits-1) - 1)
    q = clamp(round(x / scale), -2^(bits-1), 2^(bits-1) - 1)
    output = q * scale
    """
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    abs_max = np.abs(x).max()
    if abs_max < 1e-10:
        return x.copy()
    scale = abs_max / qmax
    q = np.clip(np.round(x / scale), qmin, qmax)
    return q * scale


def run_manifold_fake_quantized(points, weight_matrices, bits=8, quantize_activations=True):
    """Run manifold through network with fake quantization. (formerly run_manifold_fq)

    Simulates real integer inference:
      int8_input × int8_weight → int32_accumulator → requantize → int8_output

    The int32 accumulator means the matmul is exact — the only rounding
    happens at weight quantization and output requantization.

    Activation quantization is per-tensor: one scale for the entire
    (n_points, dim) activation matrix at each layer.
    """
    quant_weights = [fake_quantize(W, bits=bits) for W in weight_matrices]

    float_acts = [points.copy()]
    quant_acts = [points.copy()]

    val_float = points.copy()
    val_quant = points.copy()

    for W_f, W_q in zip(weight_matrices, quant_weights):
        val_float = val_float @ W_f.T
        float_acts.append(val_float.copy())

        val_quant = val_quant @ W_q.T
        if quantize_activations:
            val_quant = fake_quantize(val_quant, bits=bits)
        quant_acts.append(val_quant.copy())

    errors = [q - f for f, q in zip(float_acts, quant_acts)]
    return float_acts, quant_acts, errors


# Weight matrices (same as weight-only notebook experiment 3)
FQ_WEIGHTS = [
    np.array([[0.9, 0.2], [0.1, 1.0]]),
    np.array([[0.95, -0.15], [0.2, 0.85]]),
    np.array([[1.0, 0.1], [-0.1, 0.9]]),
    np.array([[0.85, 0.15], [0.1, 1.05]]),
]

print("Setup complete. Weight matrices loaded.")
for i, W in enumerate(FQ_WEIGHTS):
    W_q = fake_quantize(W, bits=8)
    print(f"  Layer {i+1} weight max quant error: {np.abs(W - W_q).max():.6f}")


# %%
# ============================================================
# Visualizations
# ============================================================


def plot_weight_vs_full_quantization(weight_matrices, bits=8,
                                      manifold_name='circle', n_points=128):
    """Weight-only vs weight+activation errors side-by-side. (formerly plot_weight_vs_full_comparison)"""
    points, metadata = make_manifold(manifold_name, n_points=n_points)
    t = np.linspace(0, 1, len(points))

    _, _, errors_w = run_manifold_fake_quantized(
        points, weight_matrices, bits, quantize_activations=False
    )
    _, _, errors_full = run_manifold_fake_quantized(
        points, weight_matrices, bits, quantize_activations=True
    )

    n_layers = len(weight_matrices)
    fig, axes = plt.subplots(2, n_layers + 1,
                              figsize=(3.5 * (n_layers + 1), 7))

    for row, (errors, label) in enumerate([
        (errors_w, 'Weight-only'),
        (errors_full, 'Weight + activation'),
    ]):
        ax = axes[row, 0]
        ax.scatter(points[:, 0], points[:, 1], c=t, cmap='viridis',
                   s=15, edgecolors='none')
        is_closed = metadata['type'] == 'closed'
        is_connected = metadata['type'] in ('closed', 'open')
        if is_connected:
            conn = np.vstack([points, points[0]]) if is_closed else points
            ax.plot(conn[:, 0], conn[:, 1], 'k-', alpha=0.15, linewidth=0.5)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        ax.axhline(0, color='k', linewidth=0.3)
        ax.axvline(0, color='k', linewidth=0.3)
        ax.set_ylabel(label, fontsize=10, fontweight='bold')
        if row == 0:
            ax.set_title('Input', fontweight='bold')

        for col in range(n_layers):
            ax = axes[row, col + 1]
            err = errors[col + 1]
            ax.scatter(err[:, 0], err[:, 1], c=t, cmap='viridis',
                       s=15, edgecolors='none')
            if is_connected:
                conn_e = np.vstack([err, err[0]]) if is_closed else err
                ax.plot(conn_e[:, 0], conn_e[:, 1], 'k-',
                        alpha=0.15, linewidth=0.5)
            ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
            ax.axhline(0, color='k', linewidth=0.3)
            ax.axvline(0, color='k', linewidth=0.3)
            if row == 0:
                ax.set_title(f'Error after L{col+1}', fontweight='bold')
            err_mag = np.linalg.norm(err, axis=1)
            ax.text(0.95, 0.95, f'max|e|={err_mag.max():.4f}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=7, color='red',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.8))

    plt.suptitle(
        f'{bits}-bit: Weight-only (top) vs Weight+Activation (bottom)\n'
        f'Activation requantization adds rounding at grid boundaries',
        fontsize=12, y=1.04
    )
    plt.tight_layout()
    plt.savefig(f'plots/fq{bits}_weight_vs_full.png', dpi=150,
                bbox_inches='tight')
    plt.show()


def plot_fq_error_evolution(weight_matrices, bits=8,
                            manifold_names=None, n_points=128):
    """Error manifold evolution: rows=manifolds, cols=layers."""
    if manifold_names is None:
        manifold_names = ['circle', 'ellipse', 'line', 'spiral', 'figure_eight']

    n_manifolds = len(manifold_names)
    n_layers = len(weight_matrices)
    n_cols = n_layers + 1

    fig, axes = plt.subplots(n_manifolds, n_cols,
                              figsize=(3.5 * n_cols, 3.5 * n_manifolds))

    for row, name in enumerate(manifold_names):
        points, metadata = make_manifold(name, n_points=n_points)
        float_acts, quant_acts, errors = run_manifold_fake_quantized(
            points, weight_matrices, bits
        )

        t = np.linspace(0, 1, len(points))
        is_connected = metadata['type'] in ('closed', 'open')
        is_closed = metadata['type'] == 'closed'

        ax = axes[row, 0]
        ax.scatter(points[:, 0], points[:, 1], c=t, cmap='viridis',
                   s=12, edgecolors='none')
        if is_connected:
            conn = np.vstack([points, points[0]]) if is_closed else points
            ax.plot(conn[:, 0], conn[:, 1], 'k-', alpha=0.15, linewidth=0.5)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        ax.axhline(0, color='k', linewidth=0.3)
        ax.axvline(0, color='k', linewidth=0.3)
        if row == 0:
            ax.set_title('Input', fontweight='bold')
        ax.set_ylabel(name, fontsize=11, fontweight='bold')

        for col in range(n_layers):
            ax = axes[row, col + 1]
            err = errors[col + 1]
            ax.scatter(err[:, 0], err[:, 1], c=t, cmap='viridis',
                       s=12, edgecolors='none')
            if is_connected:
                conn_e = np.vstack([err, err[0]]) if is_closed else err
                ax.plot(conn_e[:, 0], conn_e[:, 1], 'k-',
                        alpha=0.15, linewidth=0.5)
            ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
            ax.axhline(0, color='k', linewidth=0.3)
            ax.axvline(0, color='k', linewidth=0.3)
            if row == 0:
                ax.set_title(f'Error after L{col+1}', fontweight='bold')
            if col == n_layers - 1:
                err_mag = np.linalg.norm(err, axis=1)
                ax.text(0.95, 0.95, f'max|e|={err_mag.max():.4f}',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=7, color='red',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', alpha=0.8))

    plt.suptitle(f'{bits}-bit Fake Quantization — Error Manifold Evolution\n'
                 f'Per-tensor weight + activation quantization',
                 fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(f'plots/fq{bits}_error_evolution.png', dpi=150,
                bbox_inches='tight')
    plt.show()


def plot_fq_error_growth(weight_matrices, bits=8,
                         manifold_names=None, n_points=128):
    """Error magnitude growth across layers for each manifold."""
    if manifold_names is None:
        manifold_names = ['circle', 'ellipse', 'line', 'spiral', 'figure_eight']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name in manifold_names:
        points, _ = make_manifold(name, n_points=n_points)
        _, _, errors = run_manifold_fake_quantized(points, weight_matrices, bits)

        max_errors = [np.linalg.norm(err, axis=1).max() for err in errors]
        mean_errors = [np.linalg.norm(err, axis=1).mean() for err in errors]
        layers = list(range(len(errors)))

        axes[0].plot(layers, max_errors, 'o-', linewidth=2,
                     markersize=6, label=name)
        axes[1].plot(layers, mean_errors, 'o-', linewidth=2,
                     markersize=6, label=name)

    for ax, title in zip(axes, ['Max |error|', 'Mean |error|']):
        ax.set_xlabel('Layer')
        ax.set_ylabel(title)
        ax.set_title(f'{title} growth through layers')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(weight_matrices) + 1))
        ax.set_xticklabels(['input'] + [f'L{i+1}' for i in range(len(weight_matrices))])

    plt.suptitle(f'{bits}-bit Error Accumulation', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'plots/fq{bits}_error_growth.png', dpi=150,
                bbox_inches='tight')
    plt.show()


def plot_bitwidth_comparison(weight_matrices, manifold_name='circle',
                              bits_list=None, n_points=128):
    """Compare error manifolds at different bit widths. (formerly plot_bits_comparison)"""
    if bits_list is None:
        bits_list = [2, 4, 8]

    points, metadata = make_manifold(manifold_name, n_points=n_points)
    t = np.linspace(0, 1, len(points))
    is_closed = metadata['type'] == 'closed'
    is_connected = metadata['type'] in ('closed', 'open')

    n_bits = len(bits_list)
    n_layers = len(weight_matrices)

    fig, axes = plt.subplots(n_bits, n_layers + 1,
                              figsize=(3.5 * (n_layers + 1), 3.5 * n_bits))

    for row, bits in enumerate(bits_list):
        _, _, errors = run_manifold_fake_quantized(points, weight_matrices, bits)

        ax = axes[row, 0]
        ax.scatter(points[:, 0], points[:, 1], c=t, cmap='viridis',
                   s=12, edgecolors='none')
        if is_connected:
            conn = np.vstack([points, points[0]]) if is_closed else points
            ax.plot(conn[:, 0], conn[:, 1], 'k-', alpha=0.15, linewidth=0.5)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        ax.axhline(0, color='k', linewidth=0.3)
        ax.axvline(0, color='k', linewidth=0.3)
        ax.set_ylabel(f'{bits}-bit', fontsize=12, fontweight='bold')
        if row == 0:
            ax.set_title('Input', fontweight='bold')

        for col in range(n_layers):
            ax = axes[row, col + 1]
            err = errors[col + 1]
            ax.scatter(err[:, 0], err[:, 1], c=t, cmap='viridis',
                       s=12, edgecolors='none')
            if is_connected:
                conn_e = np.vstack([err, err[0]]) if is_closed else err
                ax.plot(conn_e[:, 0], conn_e[:, 1], 'k-',
                        alpha=0.15, linewidth=0.5)
            ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
            ax.axhline(0, color='k', linewidth=0.3)
            ax.axvline(0, color='k', linewidth=0.3)
            if row == 0:
                ax.set_title(f'Error after L{col+1}', fontweight='bold')
            err_mag = np.linalg.norm(err, axis=1)
            ax.text(0.95, 0.95, f'|e|={err_mag.max():.4f}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=7, color='red',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.8))

    plt.suptitle(f'Bit Width Comparison: {manifold_name}\n'
                 f'Lower bits → larger errors, more rounding distortion',
                 fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig('plots/fq_bits_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.show()


# ============================================================
# Run
# ============================================================

print("=" * 70)
print("FAKE QUANTIZATION — LAYER-BY-LAYER ERROR EVOLUTION")
print("=" * 70)
print()
print("Pipeline per layer:")
print("  int8_input × int8_weight → int32_accumulator (exact) → requantize → int8_output")
print("  Activation quantization: per-tensor (one scale for all points)")

# Weight-only vs full comparison
print("\n--- Weight-only vs weight+activation (8-bit, circle) ---")
plot_weight_vs_full_quantization(FQ_WEIGHTS, bits=8,
                                  manifold_name='circle', n_points=128)

# Full error evolution at 8-bit
plot_fq_error_evolution(FQ_WEIGHTS, bits=8, n_points=128)
plot_fq_error_growth(FQ_WEIGHTS, bits=8, n_points=128)

# Bit-width comparison
print("\nBit-width comparison on circle manifold:")
plot_bitwidth_comparison(FQ_WEIGHTS, manifold_name='circle',
                          bits_list=[2, 4, 8], n_points=128)


# %%

# %%

# %%
