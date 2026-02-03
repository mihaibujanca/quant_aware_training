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
# # Weight Quantization Error Geometry
#
# Visual exploration of how **weight-only** quantization error regions evolve through
# neural network layers. All experiments use 8-bit uniform quantization with
# delta = 1/(2^7) ≈ 0.0078.
#
# ## Key Findings
#
# ### 2D Experiments
# 1. **Uniform diagonal weights**: Error region is an axis-aligned rectangle that grows
#    uniformly. Relative error is identical across all channels.
# 2. **Non-uniform diagonal weights**: Error rectangle becomes non-square. Channels with
#    larger weights accumulate error faster.
# 3. **Full matrices (off-diagonal)**: Error region becomes a tilted parallelogram. SVD
#    reveals anisotropy — the condition number tells you how much worse error is in the
#    worst direction vs the best.
# 4. **Circle manifold**: Error magnitude and shape vary with input direction. Error
#    tubes around the input manifold are non-uniform.
#
# ### 3D Extensions
# 5. **Uniform 3D diagonal**: Error boxes grow uniformly, relative error per channel identical.
# 6. **Non-uniform 3D diagonal**: Channel sensitivity — some channels are 2-3x more
#    sensitive to quantization depending on the weight spectrum.
# 7. **Full 3D matrices**: Error polytope is non-axis-aligned. Bounding box efficiency
#    drops to ~40-60%, meaning axis-aligned bounds overestimate by 2x. SVD reveals
#    3 principal error directions.
#
# ### Manifold Analysis
# 8. **Linear error model**: With no activations, the error is `(Q_product - W_product) @ x`
#    — a single linear transform. Circles map to ellipses, lines to lines.
# 9. **Error correlates with L1 norm**: Error magnitude scales linearly with L1 norm of input.
# 10. **Grid heatmap**: Error isocontours reflect the L1-ball geometry rotated by the
#     error transform.
#
# ### ReLU Interaction
# 11. **Surviving points** (~25-30%): both float and quant paths positive — ReLU preserves
#     the error, which remains linearly correctable.
# 12. **Dead points** (~25-30%): both paths negative — ReLU zeros both, error is exactly 0.
# 13. **Mixed points** (~40-50%): some components surviving, some dead — partial information
#     loss, but surviving part is still linearly correctable.
# 14. **Flipped points** (<2%): quantization caused ReLU to make a different decision —
#     irreversible error. Rare because weight quant error is small vs signal magnitude.
# 15. **Dominant effect**: The geometric "chopping" by ReLU (dead/mixed regions) is far
#     more impactful than quantization-induced flips.
#
# ## Structure
# - **Cell 1**: Setup and imports
# - **Cell 2**: 2D experiments (uniform diagonal, non-uniform diagonal, full matrices, circle manifold)
# - **Cell 3**: 3D extensions (uniform, non-uniform, full matrices with SVD/bounding box analysis)
# - **Cell 4**: Manifold analysis (error across manifold types, heatmap, input→error mapping)
# - **Cell 5**: ReLU interaction (fate classification, error evolution, survival summary)

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

from aleph.qgeom import (
    LayerStats, ExperimentStats, AllExperimentStats,
    quantize,
    get_box_vertices_2d, minkowski_sum_2d, compute_polygon_area, transform_vertices,
    get_hypercube_vertices, minkowski_sum_3d,
    draw_polygon, set_fixed_scale,
    draw_box_3d, draw_vertices_and_hull_3d, draw_wireframe_box,
    run_experiment, run_all_manifolds,
    make_manifold, compute_pointwise_errors, compute_manifold_errors,
)

# Configuration for this notebook
BITS = 8
DELTA = 1.0 / (2 ** (BITS - 1))
N_LAYERS = 4

COLORS = {
    'layer1': '#1f77b4',
    'layer2': '#ff7f0e',
    'layer3': '#2ca02c',
    'layer4': '#d62728',
    'cumulative': '#e377c2',
    'input': '#17becf',
    'error_region': '#ff6b6b',
    'reference': '#888888'
}

ALL_STATS = AllExperimentStats()

print(f'Framework loaded. BITS={BITS}, DELTA={DELTA:.6f}')

# %%
# ============================================================
# 2D EXPERIMENTS
# ============================================================
#
# Exp 1: Uniform diagonal weights (baseline)
# Exp 2: Non-uniform diagonal weights (per-channel variation)
# Exp 3: Full matrices (channel mixing, rotation/shear)
# Exp 4: Multiple input points (error manifold)


# --- Error functions ---

def diagonal_error_fn(val, W, delta):
    """Error function for diagonal weights — axis-aligned box."""
    hw = (delta / 2) * np.abs(val)
    return get_box_vertices_2d(hw)


def full_matrix_error_fn(val, W, delta):
    """
    Error function for full matrices.

    Each output dim: error = sum_j W_err[i,j] * val[j]
    with W_err[i,j] independent in [-delta/2, delta/2].
    Half-width per output dim = delta/2 * L1_norm(val).
    """
    l1_norm = np.sum(np.abs(val))
    hw = (delta / 2) * l1_norm * np.ones(2)
    return get_box_vertices_2d(hw)


# --- Experiment definitions ---

def run_uniform_diagonal_2d(x_input):
    """Uniform diagonal weights. (formerly run_experiment_1)"""
    weights = [np.eye(2) * w for w in [0.9, 1.1, 0.85, 1.05]]
    stats, qw = run_experiment("Exp1: Uniform Diagonal", x_input, weights, diagonal_error_fn, DELTA, all_stats=ALL_STATS)
    print(f"Exp 1: volume={stats.cumulative_error_volume:.6f}, rel_error={stats.relative_error}")
    return stats, qw


def run_nonuniform_diagonal_2d(x_input):
    """Non-uniform diagonal weights. (formerly run_experiment_2)"""
    weights = [np.diag(d) for d in [[0.8, 1.2], [1.1, 0.7], [0.9, 1.1], [1.2, 0.8]]]
    stats, qw = run_experiment("Exp2: Non-Uniform Diagonal", x_input, weights, diagonal_error_fn, DELTA, all_stats=ALL_STATS)
    print(f"Exp 2: volume={stats.cumulative_error_volume:.6f}, rel_error={stats.relative_error}")
    return stats, qw


def run_full_matrix_2d(x_input):
    """Full matrices with off-diagonal elements. (formerly run_experiment_3)"""
    weights = [
        np.array([[0.9, 0.2], [0.1, 1.0]]),
        np.array([[0.95, -0.15], [0.2, 0.85]]),
        np.array([[1.0, 0.1], [-0.1, 0.9]]),
        np.array([[0.85, 0.15], [0.1, 1.05]]),
    ]
    stats, qw = run_experiment("Exp3: Full Matrices", x_input, weights, full_matrix_error_fn, DELTA, all_stats=ALL_STATS)
    print(f"Exp 3: volume={stats.cumulative_error_volume:.6f}, rel_error={stats.relative_error}")
    return stats, qw


def run_circle_manifold_2d(base_weights):
    """Multiple input points on a circle manifold. (formerly run_experiment_4)"""
    quant_weights = [quantize(W, DELTA) for W in base_weights]
    n_points = 32
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    radius = 20
    circle_points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

    results = []
    for x in circle_points:
        val = x.copy()
        cumulative_W = np.eye(2)
        cumulative_error_vertices = None

        for W in quant_weights:
            local_error_vertices = full_matrix_error_fn(val, W, DELTA)
            cumulative_W_after = W @ cumulative_W
            try:
                inv_W = np.linalg.inv(cumulative_W_after)
                error_vertices_input = transform_vertices(local_error_vertices, inv_W)
            except:
                error_vertices_input = local_error_vertices

            if cumulative_error_vertices is None:
                cumulative_error_vertices = error_vertices_input
            else:
                cumulative_error_vertices = minkowski_sum_2d(
                    cumulative_error_vertices, error_vertices_input
                )
            val = W @ val
            cumulative_W = cumulative_W_after

        error_magnitude = np.max(np.linalg.norm(cumulative_error_vertices, axis=1))
        results.append({
            'input': x.copy(),
            'error_vertices': cumulative_error_vertices.copy(),
            'error_magnitude': error_magnitude,
            'error_volume': compute_polygon_area(cumulative_error_vertices)
        })

    magnitudes = [r['error_magnitude'] for r in results]
    print(f"Exp 4: error range [{min(magnitudes):.4f}, {max(magnitudes):.4f}], "
          f"variation ratio {max(magnitudes)/min(magnitudes):.2f}x")
    return results, circle_points, quant_weights


# ============================================================
# Plotting functions (2D)
# ============================================================

def plot_diagonal_comparison(stats1, stats2, scale):
    """Compare uniform vs non-uniform diagonal. (formerly plot_experiment_1_2)"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row, stats, title in [(0, stats1, 'Exp 1: Uniform Diagonal'),
                               (1, stats2, 'Exp 2: Non-Uniform Diagonal')]:
        # Error region
        ax = axes[row, 0]
        draw_polygon(ax, stats.cumulative_error_vertices, COLORS['error_region'], alpha=0.4)
        set_fixed_scale(ax, scale)
        ax.set_title(f"{title}\nVolume: {stats.cumulative_error_volume:.6f}")
        ax.set_xlabel('Dim 0'); ax.set_ylabel('Dim 1')

        # Per-layer contributions
        ax = axes[row, 1]
        layers = [ls.layer_idx + 1 for ls in stats.layer_stats]
        x_pos = np.arange(len(layers))
        hw0 = [ls.error_half_widths[0] for ls in stats.layer_stats]
        hw1 = [ls.error_half_widths[1] for ls in stats.layer_stats]
        ax.bar(x_pos - 0.2, hw0, 0.4, label='Dim 0', color=COLORS['layer1'])
        ax.bar(x_pos + 0.2, hw1, 0.4, label='Dim 1', color=COLORS['layer2'])
        ax.set_xticks(x_pos); ax.set_xticklabels([f'L{l}' for l in layers])
        ax.set_ylabel('Error half-width'); ax.set_title(f'{title}: Per-layer error')
        ax.legend(); ax.grid(True, alpha=0.3)

        # Relative error
        ax = axes[row, 2]
        ax.bar(['Dim 0', 'Dim 1'], stats.relative_error * 100,
               color=[COLORS['layer1'], COLORS['layer2']])
        ax.set_ylabel('Relative error (%)')
        subtitle = '(Equal for uniform)' if row == 0 else '(Different per channel!)'
        ax.set_title(f'{title}: Relative error\n{subtitle}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/exp1_2_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_full_matrix_analysis(stats3, stats1, scale):
    """Full matrix error analysis with SVD. (formerly plot_experiment_3)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Error region comparison
    ax = axes[0]
    draw_polygon(ax, stats1.cumulative_error_vertices, COLORS['layer1'], alpha=0.3, label='Diagonal')
    draw_polygon(ax, stats3.cumulative_error_vertices, COLORS['error_region'], alpha=0.4, label='Full')
    set_fixed_scale(ax, scale)
    ax.set_title('Error region comparison'); ax.legend()

    # SVD shape analysis
    ax = axes[1]
    centered = stats3.cumulative_error_vertices - stats3.cumulative_error_vertices.mean(axis=0)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    ax.bar(['PC1', 'PC2'], S, color=[COLORS['layer1'], COLORS['layer2']])
    ax.set_ylabel('Singular value')
    ax.set_title(f'Error region shape (SVD)\nCondition: {S[0]/S[1]:.2f}')
    ax.grid(True, alpha=0.3)

    # Principal directions
    ax = axes[2]
    center = stats3.cumulative_error_vertices.mean(axis=0)
    draw_polygon(ax, stats3.cumulative_error_vertices, COLORS['error_region'], alpha=0.3)
    for i, (s, v) in enumerate(zip(S, Vt)):
        ax.arrow(center[0], center[1], v[0]*s*0.8, v[1]*s*0.8,
                head_width=scale*0.03, color=['blue', 'green'][i], linewidth=2,
                label=f'PC{i+1}: [{v[0]:.2f}, {v[1]:.2f}]')
    set_fixed_scale(ax, scale)
    ax.set_title('Principal directions\n(Error is anisotropic)')
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig('plots/exp3_full_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_circle_error_analysis(results, circle_points, scale):
    """Circle manifold error analysis. (formerly plot_experiment_4)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    magnitudes = [r['error_magnitude'] for r in results]

    # Error magnitude around the circle
    ax = axes[0, 0]
    scatter = ax.scatter(circle_points[:, 0], circle_points[:, 1],
                        c=magnitudes, cmap='hot', s=100, edgecolors='black')
    plt.colorbar(scatter, ax=ax, label='Error magnitude')
    ax.plot(circle_points[:, 0], circle_points[:, 1], 'b-', alpha=0.3)
    ax.set_title('Circle manifold colored by error')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # Error magnitude vs angle
    ax = axes[0, 1]
    angles = np.arctan2(circle_points[:, 1], circle_points[:, 0])
    ax.plot(np.degrees(angles), magnitudes, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Angle (degrees)'); ax.set_ylabel('Error magnitude')
    ax.set_title('Error varies with direction'); ax.grid(True, alpha=0.3)

    # Selected error regions
    ax = axes[1, 0]
    n_selected = 8
    indices = np.linspace(0, len(results)-1, n_selected, dtype=int)
    colors_sel = plt.cm.hsv(np.linspace(0, 1, n_selected))
    for idx, color in zip(indices, colors_sel):
        draw_polygon(ax, results[idx]['error_vertices'], color, alpha=0.3, linewidth=1)
    set_fixed_scale(ax, scale)
    ax.set_title('Error regions for 8 points around circle')

    # Manifold with error regions overlaid
    ax = axes[1, 1]
    error_scale = 0.5
    for r in results[::2]:
        vertices = r['error_vertices'] * error_scale + r['input']
        draw_polygon(ax, vertices, COLORS['error_region'], alpha=0.2, linewidth=0.5)
    ax.plot(circle_points[:, 0], circle_points[:, 1], 'b-', linewidth=2, label='Input manifold')
    ax.scatter(circle_points[:, 0], circle_points[:, 1], c='blue', s=20, zorder=5)
    ax.set_title('Input manifold with error \"tubes\"')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend()

    plt.tight_layout()
    plt.savefig('plots/exp4_error_manifold.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_weight_experiments_summary(all_stats):
    """Summary comparison of all weight experiments. (formerly plot_summary)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    exp_names = list(all_stats.experiments.keys())[:3]

    # Volume comparison
    ax = axes[0]
    volumes = [all_stats.experiments[name].cumulative_error_volume for name in exp_names]
    ax.bar(range(len(volumes)), volumes, color=[COLORS['layer1'], COLORS['layer2'], COLORS['layer3']])
    ax.set_xticks(range(len(volumes)))
    ax.set_xticklabels(['Uniform\nDiagonal', 'Non-Uniform\nDiagonal', 'Full\nMatrices'], fontsize=9)
    ax.set_ylabel('Error region volume')
    ax.set_title('Total error volume comparison')
    ax.grid(True, alpha=0.3)

    # Overlay all error regions
    ax = axes[1]
    colors = [COLORS['layer1'], COLORS['layer2'], COLORS['layer3']]
    alphas = [0.4, 0.3, 0.2]
    max_extent = 0
    for name, color, alpha in zip(exp_names, colors, alphas):
        stats = all_stats.experiments[name]
        draw_polygon(ax, stats.cumulative_error_vertices, color, alpha=alpha,
                    label=name.split(':')[1].strip())
        max_extent = max(max_extent, np.abs(stats.cumulative_error_vertices).max())
    set_fixed_scale(ax, max_extent * 1.2)
    ax.set_title('All error regions overlaid')
    ax.legend(loc='upper left', fontsize=8)

    # Spectral norms
    ax = axes[2]
    x_pos = np.arange(N_LAYERS)
    width = 0.25
    for i, (name, color) in enumerate(zip(exp_names, colors)):
        stats = all_stats.experiments[name]
        norms = [ls.spectral_norm for ls in stats.layer_stats]
        ax.bar(x_pos + i*width, norms, width, label=name.split(':')[1].strip(),
               color=color, alpha=0.7)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([f'L{i+1}' for i in range(N_LAYERS)])
    ax.set_ylabel('Spectral norm')
    ax.set_title('Weight spectral norms by layer')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/all_experiments_summary.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Run 2D experiments
# ============================================================

x_input = np.array([10.0, 20.0])

print("=" * 70)
print(f"2D EXPERIMENTS — Input: {x_input}, Bits: {BITS}, Delta: {DELTA}")
print("=" * 70)

stats1, weights1 = run_uniform_diagonal_2d(x_input)
stats2, weights2 = run_nonuniform_diagonal_2d(x_input)
stats3, weights3 = run_full_matrix_2d(x_input)

# Global scale for consistent 2D plots
GLOBAL_ERROR_SCALE = max(
    np.abs(s.cumulative_error_vertices).max()
    for s in [stats1, stats2, stats3]
) * 1.3

exp3_weights = [
    np.array([[0.9, 0.2], [0.1, 1.0]]),
    np.array([[0.95, -0.15], [0.2, 0.85]]),
    np.array([[1.0, 0.1], [-0.1, 0.9]]),
    np.array([[0.85, 0.15], [0.1, 1.05]]),
]
results4, circle_points, weights4 = run_circle_manifold_2d(exp3_weights)

# Plot
plot_diagonal_comparison(stats1, stats2, GLOBAL_ERROR_SCALE)
plot_full_matrix_analysis(stats3, stats1, GLOBAL_ERROR_SCALE)
plot_circle_error_analysis(results4, circle_points, GLOBAL_ERROR_SCALE)
plot_weight_experiments_summary(ALL_STATS)
ALL_STATS.print_summary()

print("\nKEY TAKEAWAYS:")
print("1. Uniform diagonal: relative error constant across channels")
print("2. Non-uniform diagonal: channels accumulate error at different rates")
print("3. Full matrices: error region tilted/sheared, PCA reveals anisotropy")
print("4. Error manifold: magnitude and shape vary with input position and direction")


# %%
# ============================================================
# 3D EXTENSIONS
# ============================================================
#
# Same 3 experiments but in 3D, revealing:
#   - 3D box/parallelepiped geometry
#   - Channel sensitivity analysis
#   - SVD shape analysis with 3 singular values
#   - Bounding box efficiency
#   - 2D projections

np.random.seed(42)

x_input_3d = np.array([10.0, 20.0, 30.0])


# ============================================================
# 3D error computation
# ============================================================

def compute_diagonal_error_boxes_3d(x, weight_channels, delta=DELTA):
    """
    Compute per-layer error boxes for diagonal weight networks (3D). (formerly compute_error_boxes_diagonal_3d)

    Args:
        x: Input vector (3D)
        weight_channels: (n_layers, 3) array of per-channel weights
    Returns:
        List of per-layer error box info
    """
    quant_w = quantize(weight_channels, delta)
    boxes = []
    val = x.copy()
    cumulative_weight = np.ones(3)

    for i in range(len(quant_w)):
        w = quant_w[i]
        box_at_layer = (delta / 2) * np.abs(val)
        cumulative_weight_after = cumulative_weight * w
        box_in_input_space = box_at_layer / np.abs(cumulative_weight_after)

        boxes.append({
            'layer': i + 1,
            'hw_input': box_in_input_space.copy(),
            'hw_output': box_at_layer.copy(),
            'value': val.copy(),
            'weight': w.copy(),
        })
        val = w * val
        cumulative_weight = cumulative_weight_after

    return boxes, quant_w


def trace_full_matrix_error_3d(x, quant_weights, delta=DELTA):
    """
    Trace error region geometry through 3D full-matrix layers. (formerly trace_error_geometry_3d)

    At each layer, the error from weight quantization is an axis-aligned box
    in output space. Mapped back to input space and accumulated via
    Minkowski sum, the cumulative error becomes a polytope.
    """
    history = []
    val = x.copy()
    cumulative_transform = np.eye(3)

    for i, W in enumerate(quant_weights):
        l1_norm = np.sum(np.abs(val))
        hw = delta / 2 * l1_norm * np.ones(3)
        local_vertices = get_hypercube_vertices(1.0, dims=3) * hw

        cumulative_transform_after = W @ cumulative_transform
        try:
            inv_t = np.linalg.inv(cumulative_transform_after)
            error_vertices_input = local_vertices @ inv_t.T
        except np.linalg.LinAlgError:
            error_vertices_input = local_vertices

        if i == 0:
            total_vertices = error_vertices_input
        else:
            total_vertices = minkowski_sum_3d(total_vertices, error_vertices_input)

        history.append({
            'layer': i + 1,
            'value': val.copy(),
            'error_vertices_input': error_vertices_input.copy(),
            'cumulative_vertices': total_vertices.copy(),
            'hw_local': hw.copy(),
        })

        val = W @ val
        cumulative_transform = cumulative_transform_after

    return history


# ============================================================
# Exp 1 (3D): Uniform diagonal — cumulative error boxes
# ============================================================

print("=" * 70)
print(f"3D EXPERIMENTS — Input: {x_input_3d}, Bits: {BITS}")
print("=" * 70)

uniform_weights_3d = np.array([[0.8, 0.8, 0.8],
                                [1.2, 1.2, 1.2],
                                [0.9, 0.9, 0.9],
                                [1.1, 1.1, 1.1]])

boxes_uniform, qw_uniform = compute_diagonal_error_boxes_3d(x_input_3d, uniform_weights_3d)

total_hw_uniform = sum(b['hw_input'] for b in boxes_uniform)
print(f"\nExp 1 (3D Uniform): total half-widths = {total_hw_uniform}")
print(f"Relative error per channel: {total_hw_uniform / x_input_3d * 100}%")
print("(Same % for all channels — uniform weights)")

# Plot: nested cumulative error boxes + error growth
fig = plt.figure(figsize=(14, 5))

# Cumulative Minkowski sum (nested boxes)
ax1 = fig.add_subplot(131, projection='3d')
colors_3d = plt.cm.viridis(np.linspace(0.2, 0.8, len(boxes_uniform)))
cumulative_hw = np.zeros(3)
for b, color in zip(boxes_uniform, colors_3d):
    cumulative_hw = cumulative_hw + b['hw_input']
    draw_box_3d(ax1, np.zeros(3), cumulative_hw, color, alpha=0.2)
ax1.set_xlabel('Ch0 (x=10)'); ax1.set_ylabel('Ch1 (x=20)'); ax1.set_zlabel('Ch2 (x=30)')
ax1.set_title('Cumulative error box\n(Minkowski sum, nested)')

# Final box with axis lines
ax2 = fig.add_subplot(132, projection='3d')
draw_box_3d(ax2, np.zeros(3), total_hw_uniform, 'red', alpha=0.4)
for i, (hw, c, label) in enumerate(zip(total_hw_uniform, ['b', 'g', 'r'],
                                        ['Ch0 (x=10)', 'Ch1 (x=20)', 'Ch2 (x=30)'])):
    pts = np.zeros((2, 3))
    pts[0, i] = -hw; pts[1, i] = hw
    ax2.plot3D(pts[:, 0], pts[:, 1], pts[:, 2], f'{c}-', linewidth=3,
              label=f'{label}: ±{hw:.4f}')
ax2.set_title('Final error box\nLarger input → larger error')
ax2.legend(loc='upper left', fontsize=7)

# Error growth per channel
ax3 = fig.add_subplot(133)
for ch in range(3):
    cumulative = np.cumsum([b['hw_input'][ch] for b in boxes_uniform])
    percentage = 100 * cumulative / x_input_3d[ch]
    ax3.plot([b['layer'] for b in boxes_uniform], percentage, 'o-',
             linewidth=2, markersize=8, label=f'Ch{ch}')
ax3.set_xlabel('Layer'); ax3.set_ylabel('Error as % of input')
ax3.set_title('Relative error (same for all channels)')
ax3.legend(); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/exp1_3d_uniform.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# Exp 2 (3D): Non-uniform diagonal — channel sensitivity
# ============================================================

nonuniform_weights_3d = np.array([
    [0.8, 1.2, 0.5],   # Ch1 amplified, Ch2 shrunk
    [1.1, 0.7, 1.3],   # Ch0,2 amplified, Ch1 shrunk
    [0.9, 1.1, 0.9],   # Mild
    [1.2, 0.8, 1.1],   # Ch0 amplified, Ch1 shrunk
])

boxes_nonuniform, qw_nonuniform = compute_diagonal_error_boxes_3d(x_input_3d, nonuniform_weights_3d)
total_hw_nonuniform = sum(b['hw_input'] for b in boxes_nonuniform)

print(f"\nExp 2 (3D Non-Uniform): total half-widths = {total_hw_nonuniform}")

fig = plt.figure(figsize=(18, 5))

# Overlay both boxes
ax1 = fig.add_subplot(131, projection='3d')
max_hw = max(total_hw_nonuniform.max(), total_hw_uniform.max())
draw_box_3d(ax1, np.zeros(3), total_hw_nonuniform, 'red', alpha=0.3)
draw_box_3d(ax1, np.zeros(3), total_hw_uniform, 'blue', alpha=0.3)
ax1.set_xlabel('Ch0'); ax1.set_ylabel('Ch1'); ax1.set_zlabel('Ch2')
ax1.set_title('Overlay\nRed=Non-uniform, Blue=Uniform')
ax1.set_xlim(-max_hw*1.2, max_hw*1.2)
ax1.set_ylim(-max_hw*1.2, max_hw*1.2)
ax1.set_zlim(-max_hw*1.2, max_hw*1.2)

# Relative error comparison
ax2 = fig.add_subplot(132)
uniform_rel = total_hw_uniform / x_input_3d * 100
nonuniform_rel = total_hw_nonuniform / x_input_3d * 100
x_pos = np.arange(3)
ax2.bar(x_pos - 0.2, uniform_rel, 0.4, label='Uniform', color='blue', alpha=0.7)
ax2.bar(x_pos + 0.2, nonuniform_rel, 0.4, label='Non-uniform', color='red', alpha=0.7)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'Ch{i} (x={int(x_input_3d[i])})' for i in range(3)])
ax2.set_ylabel('Relative error (%)')
ax2.set_title('Relative error comparison')
ax2.legend(); ax2.grid(True, alpha=0.3)

# Channel sensitivity
ax3 = fig.add_subplot(133)
mean_rel = nonuniform_rel.mean()
sensitivity = nonuniform_rel / mean_rel
ax3.bar(x_pos, sensitivity, color=['green' if s < 1 else 'red' for s in sensitivity], alpha=0.7)
ax3.axhline(1.0, color='gray', linestyle='--', linewidth=2, label='Average')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f'Ch{i}' for i in range(3)])
ax3.set_ylabel('Sensitivity (relative to mean)')
ax3.set_title('Channel error sensitivity\nGreen=below avg, Red=above avg')
for i, s in enumerate(sensitivity):
    ax3.annotate(f'{s:.2f}x', (i, s + 0.05), ha='center', fontsize=12)
ax3.legend(); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/exp2_3d_sensitivity.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# Exp 3 (3D): Full matrices — parallelepipeds, SVD, projections
# ============================================================

full_weights_3d = [
    np.array([[0.9, 0.1, 0.0], [0.1, 1.1, 0.1], [0.0, 0.1, 0.8]]),
    np.array([[0.8, -0.3, 0.1], [0.3, 0.8, -0.2], [-0.1, 0.2, 0.9]]),
    np.array([[1.1, 0.2, 0.0], [0.0, 0.9, 0.2], [0.1, 0.0, 1.0]]),
    np.array([[0.9, 0.2, -0.1], [-0.2, 1.0, 0.1], [0.1, -0.1, 0.85]]),
]

quant_full_3d = [quantize(W, DELTA) for W in full_weights_3d]
history_3d = trace_full_matrix_error_3d(x_input_3d, quant_full_3d)

# Also compute diagonal-only case for comparison
diagonal_3d = [np.diag(np.diag(W)) for W in quant_full_3d]
history_diag_3d = trace_full_matrix_error_3d(x_input_3d, diagonal_3d)

final_verts = history_3d[-1]['cumulative_vertices']
final_verts_diag = history_diag_3d[-1]['cumulative_vertices']

print(f"\nExp 3 (3D Full Matrices):")
print(f"  Final error vertices: {len(final_verts)}")
print(f"  Bounding box: {final_verts.min(axis=0)} to {final_verts.max(axis=0)}")

# Figure 1: Error region evolution
fig = plt.figure(figsize=(18, 5))
for i, h in enumerate(history_3d):
    ax = fig.add_subplot(1, 4, i+1, projection='3d')
    draw_vertices_and_hull_3d(ax, h['error_vertices_input'], colors_3d[i], alpha=0.4)
    draw_vertices_and_hull_3d(ax, h['cumulative_vertices'], 'red', alpha=0.2)
    me = np.abs(h['cumulative_vertices']).max() * 1.2
    ax.set_xlim(-me, me); ax.set_ylim(-me, me); ax.set_zlim(-me, me)
    ax.set_xlabel('Ch0'); ax.set_ylabel('Ch1'); ax.set_zlabel('Ch2')
    ax.set_title(f"Layer {h['layer']}")
plt.suptitle('3D Error Region Evolution (green=layer contribution, red=cumulative)', fontsize=12)
plt.tight_layout()
plt.savefig('plots/exp3_3d_evolution.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 2: Final region vs bounding box
fig = plt.figure(figsize=(16, 6))
bbox_hw = (final_verts.max(axis=0) - final_verts.min(axis=0)) / 2
max_extent = np.abs(final_verts).max() * 1.2

ax1 = fig.add_subplot(131, projection='3d')
draw_vertices_and_hull_3d(ax1, final_verts, 'red', alpha=0.4)
ax1.set_xlim(-max_extent, max_extent); ax1.set_ylim(-max_extent, max_extent)
ax1.set_zlim(-max_extent, max_extent)
ax1.set_title('Actual error region\n(Non-axis-aligned polytope)')

ax2 = fig.add_subplot(132, projection='3d')
box_verts = get_hypercube_vertices(1.0) * bbox_hw
draw_vertices_and_hull_3d(ax2, box_verts, 'blue', alpha=0.4)
ax2.set_xlim(-max_extent, max_extent); ax2.set_ylim(-max_extent, max_extent)
ax2.set_zlim(-max_extent, max_extent)
ax2.set_title('Bounding box\n(Axis-aligned approximation)')

ax3 = fig.add_subplot(133, projection='3d')
draw_vertices_and_hull_3d(ax3, final_verts, 'red', alpha=0.3)
draw_wireframe_box(ax3, bbox_hw, 'blue', alpha=0.8)
ax3.set_xlim(-max_extent, max_extent); ax3.set_ylim(-max_extent, max_extent)
ax3.set_zlim(-max_extent, max_extent)
ax3.set_title('Overlay\nRed=Actual, Blue=Bounding box')
for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('Ch0'); ax.set_ylabel('Ch1'); ax.set_zlabel('Ch2')

plt.tight_layout()
plt.savefig('plots/exp3_3d_vs_bbox.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 3: SVD, volume growth, bounding box efficiency, 2D projections
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# SVD of final error region
centered = final_verts - final_verts.mean(axis=0)
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
centered_diag = final_verts_diag - final_verts_diag.mean(axis=0)
U_d, S_d, Vt_d = np.linalg.svd(centered_diag, full_matrices=False)

ax = axes[0, 0]
x_pos = np.arange(3)
ax.bar(x_pos - 0.175, S, 0.35, label='Full matrix', color='red', alpha=0.7)
ax.bar(x_pos + 0.175, S_d, 0.35, label='Diagonal only', color='blue', alpha=0.7)
ax.set_xticks(x_pos); ax.set_xticklabels(['PC1', 'PC2', 'PC3'])
ax.set_ylabel('Singular value')
ax.set_title(f'Error shape: Full vs Diagonal\nCondition: {S[0]/S[2]:.2f} vs {S_d[0]/S_d[2]:.2f}')
ax.legend(); ax.grid(True, alpha=0.3)

# Volume growth
ax = axes[0, 1]
volumes_full = []
volumes_diag = []
for h_f, h_d in zip(history_3d, history_diag_3d):
    try: volumes_full.append(ConvexHull(h_f['cumulative_vertices']).volume)
    except: volumes_full.append(0)
    try: volumes_diag.append(ConvexHull(h_d['cumulative_vertices']).volume)
    except: volumes_diag.append(0)

layers = [h['layer'] for h in history_3d]
ax.plot(layers, volumes_full, 'o-', linewidth=2, label='Full matrix', color='red')
ax.plot(layers, volumes_diag, 's-', linewidth=2, label='Diagonal only', color='blue')
ax.set_xlabel('Layer'); ax.set_ylabel('Error region volume')
ax.set_title('Error volume growth'); ax.legend(); ax.grid(True, alpha=0.3)

# Bounding box efficiency
ax = axes[1, 0]
efficiencies = []
for h in history_3d:
    v = h['cumulative_vertices']
    bbox_vol = np.prod(v.max(axis=0) - v.min(axis=0))
    try: actual_vol = ConvexHull(v).volume
    except: actual_vol = bbox_vol
    efficiencies.append(actual_vol / bbox_vol if bbox_vol > 0 else 1)

ax.bar(layers, efficiencies, color='purple', alpha=0.7)
ax.axhline(1.0, color='gray', linestyle='--', label='Perfect efficiency (cube)')
ax.set_xlabel('Layer'); ax.set_ylabel('Actual / Bounding box volume')
ax.set_title(f'Bounding box efficiency\nFinal: {efficiencies[-1]:.3f} '
             f'(overestimates by {100*(1/efficiencies[-1]-1):.0f}%)')
ax.set_ylim(0, 1.2); ax.legend(); ax.grid(True, alpha=0.3)

# 2D projections
ax = axes[1, 1]
projections = [(0, 1, 'Ch0-Ch1'), (0, 2, 'Ch0-Ch2'), (1, 2, 'Ch1-Ch2')]
proj_colors = [COLORS['layer1'], COLORS['layer2'], COLORS['layer3']]
for (i, j, name), color in zip(projections, proj_colors):
    proj_full = final_verts[:, [i, j]]
    proj_diag = final_verts_diag[:, [i, j]]
    try:
        hull = ConvexHull(proj_full)
        hv = proj_full[hull.vertices]
        hv = np.vstack([hv, hv[0]])
        ax.fill(hv[:, 0], hv[:, 1], color=color, alpha=0.2, label=f'Full {name}')
        ax.plot(hv[:, 0], hv[:, 1], color=color, linewidth=2)
    except: pass
    try:
        hull_d = ConvexHull(proj_diag)
        hv_d = proj_diag[hull_d.vertices]
        hv_d = np.vstack([hv_d, hv_d[0]])
        ax.plot(hv_d[:, 0], hv_d[:, 1], color=color, linewidth=1, linestyle='--')
    except: pass
ax.set_xlabel('Dim A'); ax.set_ylabel('Dim B')
ax.set_title('2D projections of 3D error\n(Solid=full, Dashed=diagonal)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('plots/exp3_3d_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n3D SVD principal directions:")
for i, v in enumerate(Vt):
    print(f"  PC{i+1} (σ={S[i]:.6f}): [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]")
print(f"\nFull matrix volume: {volumes_full[-1]:.6f}")
print(f"Diagonal-only volume: {volumes_diag[-1]:.6f}")
print(f"Ratio: {volumes_full[-1]/volumes_diag[-1]:.3f}x")


# %%
# ============================================================
# MANIFOLD ANALYSIS
# ============================================================
#
# How does quantization error vary across different input manifolds?
# Uses functions from qgeom.manifolds (make_manifold, compute_pointwise_errors,
# compute_manifold_errors) and qgeom.experiment (run_all_manifolds).
#
# Key insight: since the network is linear (no activations), the error is a
# linear function of input: error(x) = (Q_product - W_product) @ x.
# So circles map to ellipses, lines to lines, etc.


def _square_lims(ax, margin=1.05):
    """Equalize axis ranges to make subplot square, centered on data midpoint."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cx, cy = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
    half = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2 * margin
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


# Manifold visualization
# ============================================================

def plot_manifold_comparison(all_results, scale=None):
    """Compare error patterns across manifolds."""
    n_manifolds = len(all_results)
    n_cols = 3
    n_rows = (n_manifolds + n_cols - 1) // n_cols

    if scale is None:
        scale = max(np.abs(d['points']).max() for d in all_results.values()) * 1.4

    # Figure 1: Manifolds colored by error
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()

    for idx, (name, data) in enumerate(all_results.items()):
        ax = axes[idx]
        points = data['points']
        magnitudes = [r['error_magnitude'] for r in data['results']]

        scatter = ax.scatter(points[:, 0], points[:, 1],
                            c=magnitudes, cmap='hot', s=60, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Error mag')

        if data['metadata']['type'] in ('closed', 'open'):
            conn = np.vstack([points, points[0]]) if data['metadata']['type'] == 'closed' else points
            ax.plot(conn[:, 0], conn[:, 1], 'b-', alpha=0.3, linewidth=1)

        stats = data['stats']
        ax.set_title(f"{name}\nVar ratio: {stats['variation_ratio']:.2f}x, "
                    f"Corr(L1): {stats['correlation_l1']:.2f}")
        set_fixed_scale(ax, scale)

    for idx in range(len(all_results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('plots/manifolds_error_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Figure 2: Summary statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = list(all_results.keys())
    x_pos = np.arange(len(names))

    # Error range
    ax = axes[0]
    maxs = [all_results[n]['stats']['error_mag_max'] for n in names]
    means = [all_results[n]['stats']['error_mag_mean'] for n in names]
    mins = [all_results[n]['stats']['error_mag_min'] for n in names]
    ax.bar(x_pos, maxs, alpha=0.3, color='red', label='Max')
    ax.bar(x_pos, means, alpha=0.5, color='blue', label='Mean')
    ax.bar(x_pos, mins, alpha=0.7, color='green', label='Min')
    ax.set_xticks(x_pos); ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Error magnitude')
    ax.set_title('Error range by manifold'); ax.legend(); ax.grid(True, alpha=0.3)

    # Variation ratio
    ax = axes[1]
    ratios = [all_results[n]['stats']['variation_ratio'] for n in names]
    ax.bar(x_pos, ratios, color='purple', alpha=0.7)
    ax.set_xticks(x_pos); ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Max/Min error ratio')
    ax.set_title('Error variation within manifold'); ax.axhline(1.0, color='gray', linestyle='--')
    ax.grid(True, alpha=0.3)

    # Correlation with L1 norm
    ax = axes[2]
    corrs = [all_results[n]['stats']['correlation_l1'] for n in names]
    colors = ['green' if c > 0.8 else 'orange' if c > 0.5 else 'red' for c in corrs]
    ax.bar(x_pos, corrs, color=colors, alpha=0.7)
    ax.set_xticks(x_pos); ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Correlation')
    ax.set_title('Error vs L1 norm correlation\n(Green=predictable, Red=complex)')
    ax.set_ylim(0, 1.1); ax.axhline(1.0, color='gray', linestyle='--'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/manifolds_statistics.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_error_heatmap(quant_weights, delta=DELTA, extent=30, n_side=15):
    """Error magnitude heatmap across input space. (formerly plot_grid_heatmap)"""
    grid_1d = np.linspace(-extent, extent, n_side)
    grid_x, grid_y = np.meshgrid(grid_1d, grid_1d)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    results = compute_manifold_errors(grid_points, quant_weights, delta)
    errors = np.array([r['error_magnitude'] for r in results]).reshape(n_side, n_side)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap
    ax = axes[0]
    im = ax.imshow(errors, extent=[-extent, extent, -extent, extent], origin='lower', cmap='hot')
    plt.colorbar(im, ax=ax, label='Error magnitude')
    ax.set_xlabel('Input dim 0'); ax.set_ylabel('Input dim 1')
    ax.set_title('Error magnitude across input space')
    ax.set_aspect('equal')

    # Error vs L1 norm
    ax = axes[1]
    l1_norms = np.sum(np.abs(grid_points), axis=1)
    error_flat = errors.ravel()
    ax.scatter(l1_norms, error_flat, alpha=0.5, s=20,
              c=np.arctan2(grid_points[:, 1], grid_points[:, 0]), cmap='hsv')
    z = np.polyfit(l1_norms, error_flat, 1)
    x_fit = np.linspace(l1_norms.min(), l1_norms.max(), 100)
    ax.plot(x_fit, np.poly1d(z)(x_fit), 'r-', linewidth=2,
           label=f'Linear fit: y={z[0]:.4f}x + {z[1]:.4f}')
    ax.set_xlabel('L1 norm of input'); ax.set_ylabel('Error magnitude')
    ax.set_title('Error scales with L1 norm\n(color=angle, shows directional dependence)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/manifold_grid_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_error_manifolds(true_weights, quant_weights, manifold_names=None, n_points=64):
    """Plot input manifold vs error manifold side by side.

    Since the network is linear (no activations), the error is:
        error(x) = (Q_n...Q_1 - W_n...W_1) @ x
    This is a linear map, so circles become ellipses, lines stay lines, etc.

    Color encodes position along the manifold so you can see which input
    point maps to which error point.
    """
    if manifold_names is None:
        manifold_names = ['circle', 'ellipse', 'line', 'spiral', 'figure_eight']

    n = len(manifold_names)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))

    for i, name in enumerate(manifold_names):
        points, metadata = make_manifold(name, n_points=n_points)
        errors, W_error = compute_pointwise_errors(points, true_weights, quant_weights)

        # Color by position along manifold
        t = np.linspace(0, 1, len(points))

        # --- Input manifold ---
        ax = axes[i, 0]
        ax.scatter(points[:, 0], points[:, 1], c=t, cmap='viridis',
                   s=30, edgecolors='black', linewidth=0.5)
        if metadata['type'] in ('closed', 'open'):
            conn = np.vstack([points, points[0]]) if metadata['type'] == 'closed' else points
            ax.plot(conn[:, 0], conn[:, 1], 'k-', alpha=0.3, linewidth=1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5); ax.axvline(0, color='k', linewidth=0.5)
        ax.set_title(f'{name} — input')
        if i == n - 1:
            ax.set_xlabel('Dim 0')
        ax.set_ylabel('Dim 1')
        _square_lims(ax)

        # --- Error manifold ---
        ax = axes[i, 1]
        ax.scatter(errors[:, 0], errors[:, 1], c=t, cmap='viridis',
                   s=30, edgecolors='black', linewidth=0.5)
        if metadata['type'] in ('closed', 'open'):
            conn_e = np.vstack([errors, errors[0]]) if metadata['type'] == 'closed' else errors
            ax.plot(conn_e[:, 0], conn_e[:, 1], 'k-', alpha=0.3, linewidth=1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5); ax.axvline(0, color='k', linewidth=0.5)
        error_mag = np.linalg.norm(errors, axis=1)
        ax.set_title(f'{name} — error (max |e|={error_mag.max():.4f})')
        if i == n - 1:
            ax.set_xlabel('Error dim 0')
        ax.set_ylabel('Error dim 1')
        _square_lims(ax)

    plt.suptitle('Input Manifold  →  Error Manifold\n'
                 'Error is a linear map of input: circles→ellipses, lines→lines',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('plots/error_manifolds.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print the error transform properties
    _, W_error = compute_pointwise_errors(np.eye(2), true_weights, quant_weights)
    U, S, Vt = np.linalg.svd(W_error)
    print(f"\nError transform matrix (Q_product - W_product):")
    print(f"  [[{W_error[0,0]:.6f}, {W_error[0,1]:.6f}],")
    print(f"   [{W_error[1,0]:.6f}, {W_error[1,1]:.6f}]]")
    print(f"  Singular values: {S[0]:.6f}, {S[1]:.6f}")
    print(f"  Condition number: {S[0]/S[1]:.2f}")
    print(f"  → Circle of radius r maps to ellipse with semi-axes "
          f"{S[0]:.6f}r x {S[1]:.6f}r")
    print(f"  → Max stretch direction: [{Vt[0,0]:.3f}, {Vt[0,1]:.3f}]")
    print(f"  → Min stretch direction: [{Vt[1,0]:.3f}, {Vt[1,1]:.3f}]")



def plot_weight_error_evolution(true_weights, manifold_names=None, n_points=128):
    """Error manifold evolution layer by layer (weight-only).

    Shows how the error grows through layers. Since the error at each
    stage is a linear transform of input, shapes are preserved:
    circles stay ellipses, lines stay lines, etc.

    Grid: rows = manifolds, cols = input + error after each layer.
    """
    if manifold_names is None:
        manifold_names = ['circle', 'ellipse', 'line', 'spiral', 'figure_eight']

    quant_weights = [quantize(W, DELTA) for W in true_weights]
    n_manifolds = len(manifold_names)
    n_layers = len(true_weights)

    fig, axes = plt.subplots(n_manifolds, n_layers + 1,
                              figsize=(3.5 * (n_layers + 1), 3.5 * n_manifolds))

    for row, name in enumerate(manifold_names):
        points, metadata = make_manifold(name, n_points=n_points)
        t = np.linspace(0, 1, len(points))
        is_connected = metadata['type'] in ('closed', 'open')
        is_closed = metadata['type'] == 'closed'

        # Input manifold
        ax = axes[row, 0]
        ax.scatter(points[:, 0], points[:, 1], c=t, cmap='viridis',
                   s=15, edgecolors='none')
        if is_connected:
            conn = np.vstack([points, points[0]]) if is_closed else points
            ax.plot(conn[:, 0], conn[:, 1], 'k-', alpha=0.2, linewidth=0.8)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        ax.axhline(0, color='k', linewidth=0.3)
        ax.axvline(0, color='k', linewidth=0.3)
        if row == 0:
            ax.set_title('Input', fontweight='bold')
        ax.set_ylabel(name, fontsize=11, fontweight='bold')
        _square_lims(ax)

        # Error after each layer: error_k(x) = (Q_k...Q_1 - W_k...W_1) @ x
        W_float = np.eye(2)
        W_quant = np.eye(2)
        for col in range(n_layers):
            W_float = true_weights[col] @ W_float
            W_quant = quant_weights[col] @ W_quant
            W_error = W_quant - W_float
            errors = points @ W_error.T

            ax = axes[row, col + 1]
            ax.scatter(errors[:, 0], errors[:, 1], c=t, cmap='viridis',
                       s=15, edgecolors='none')
            if is_connected:
                conn_e = np.vstack([errors, errors[0]]) if is_closed else errors
                ax.plot(conn_e[:, 0], conn_e[:, 1], 'k-', alpha=0.2, linewidth=0.8)
            ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
            ax.axhline(0, color='k', linewidth=0.3)
            ax.axvline(0, color='k', linewidth=0.3)
            if row == 0:
                ax.set_title(f'Error after L{col+1}', fontweight='bold')
            err_mag = np.linalg.norm(errors, axis=1)
            ax.text(0.95, 0.95, f'max|e|={err_mag.max():.4f}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=7, color='red',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.8))
            _square_lims(ax)

    plt.suptitle('Weight-Only Error Evolution Through Layers\n'
                 'Error is always a linear transform — shapes are preserved',
                 fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig('plots/weight_error_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Run manifold analysis
# ============================================================

# Use weights from experiment 3
manifold_weights = [
    np.array([[0.9, 0.2], [0.1, 1.0]]),
    np.array([[0.95, -0.15], [0.2, 0.85]]),
    np.array([[1.0, 0.1], [-0.1, 0.9]]),
    np.array([[0.85, 0.15], [0.1, 1.05]]),
]
manifold_qw = [quantize(W, DELTA) for W in manifold_weights]

print("=" * 70)
print("MANIFOLD COMPARISON")
print("=" * 70)

all_manifold_results = run_all_manifolds(manifold_qw, n_points=48, delta=DELTA)

print(f"\n{'Manifold':<15} {'Min Error':<12} {'Max Error':<12} {'Var Ratio':<12} {'Corr(L1)':<10}")
print("-" * 60)
for name, data in all_manifold_results.items():
    s = data['stats']
    print(f"{name:<15} {s['error_mag_min']:<12.4f} {s['error_mag_max']:<12.4f} "
          f"{s['variation_ratio']:<12.2f} {s['correlation_l1']:<10.2f}")

plot_manifold_comparison(all_manifold_results)
plot_error_heatmap(manifold_qw)

# Error manifold visualization: input shape → error shape
print("\n" + "=" * 70)
print("ERROR MANIFOLDS: Input Shape → Error Shape")
print("=" * 70)
plot_error_manifolds(manifold_weights, manifold_qw)

# Layer-by-layer error evolution
print("\n" + "=" * 70)
print("WEIGHT ERROR EVOLUTION — LAYER BY LAYER")
print("=" * 70)
plot_weight_error_evolution(manifold_weights, n_points=128)

print("\nOBSERVATIONS:")
print("1. CIRCLE → ELLIPSE: the linear error transform stretches/rotates")
print("2. LINE → LINE: at a different angle and length")
print("3. SPIRAL → SPIRAL: transformed but still spiral-shaped")
print("4. The error transform is a single 2x2 matrix applied to all inputs")
print("5. SVD of the error matrix tells you: how much stretch, in what direction")
print("6. High condition number = error is much worse in some directions than others")

# %%
# ============================================================
# RELU INTERACTION WITH WEIGHT QUANTIZATION ERROR
# ============================================================
#
# How does ReLU interact with weight quantization error?
# With no activations (previous cells), error is a linear function of input.
# ReLU introduces a nonlinearity that can:
#   1. Preserve error (both paths positive → "surviving")
#   2. Zero out error (both paths negative → "dead")
#   3. Irreversibly change the computation (sign disagreement → "flipped")
#   4. Partially preserve (mixed: some components surviving, some dead)
#
# Key insight: with near-identity weights and radius-20 manifolds,
# quantization error is ~0.004 per element. Flips only happen when
# |pre_relu_value| < error, which is a tiny fraction near the axes.

# Override manifold defaults for ReLU experiments:
# The default line [-25,-10]->[25,10] stays in Q1/Q3 (both components same sign),
# so we only see surviving+dead. Use a line that crosses both axes to get mixed.
RELU_MANIFOLD_KWARGS = {
    'line': {'start': np.array([-20, 10]), 'end': np.array([20, -10])},
}


def propagate_with_relu(points, true_weights, quant_weights):
    """Propagate manifold through W @ x -> ReLU layers, tracking both paths.

    Returns list of dicts (one per layer), each containing:
      - float_pre_relu, float_post_relu: (n_points, 2)
      - quant_pre_relu, quant_post_relu: (n_points, 2)
      - error: quant_post_relu - float_post_relu
      - fate: (n_points,) array of 'surviving'/'flipped'/'dead'/'mixed'
      - stats: dict with counts and fractions
    """
    layers = []
    float_val = points.copy()
    quant_val = points.copy()

    for W_f, W_q in zip(true_weights, quant_weights):
        float_pre = float_val @ W_f.T
        quant_pre = quant_val @ W_q.T

        float_post = np.maximum(0, float_pre)
        quant_post = np.maximum(0, quant_pre)

        # Per-component classification
        float_pos = float_pre >= 0  # (n_points, 2) bool
        quant_pos = quant_pre >= 0
        comp_surviving = float_pos & quant_pos
        comp_dead = ~float_pos & ~quant_pos
        comp_flipped = float_pos != quant_pos

        # Point-level classification (worst case across components)
        any_flip = comp_flipped.any(axis=1)
        all_dead = comp_dead.all(axis=1)
        all_surviving = comp_surviving.all(axis=1)

        fate = np.full(len(points), 'mixed', dtype=object)
        fate[all_surviving] = 'surviving'
        fate[all_dead] = 'dead'
        fate[any_flip] = 'flipped'  # overrides mixed/dead if any component flipped

        error = quant_post - float_post

        n = len(points)
        stats = {
            'n_surviving': int((fate == 'surviving').sum()),
            'n_flipped': int((fate == 'flipped').sum()),
            'n_dead': int((fate == 'dead').sum()),
            'n_mixed': int((fate == 'mixed').sum()),
            'frac_surviving': (fate == 'surviving').sum() / n,
            'frac_flipped': (fate == 'flipped').sum() / n,
            'frac_dead': (fate == 'dead').sum() / n,
            'frac_mixed': (fate == 'mixed').sum() / n,
            # Component-level stats
            'n_comp_surviving': int(comp_surviving.sum()),
            'n_comp_flipped': int(comp_flipped.sum()),
            'n_comp_dead': int(comp_dead.sum()),
        }

        layers.append({
            'float_pre_relu': float_pre,
            'float_post_relu': float_post,
            'quant_pre_relu': quant_pre,
            'quant_post_relu': quant_post,
            'error': error,
            'fate': fate,
            'stats': stats,
        })

        float_val = float_post
        quant_val = quant_post

    return layers


def plot_relu_error_evolution(true_weights, manifold_names=None, n_points=128):
    """Grid of manifolds x layers showing transformed data and error.

    For each manifold, two rows:
      Row 1 (data): input manifold + float post-ReLU transformed manifold per layer
      Row 2 (error): error colored by ReLU fate per layer
    """
    if manifold_names is None:
        manifold_names = ['circle', 'ellipse', 'line', 'spiral', 'figure_eight']

    quant_weights = [quantize(W, DELTA) for W in true_weights]
    n_manifolds = len(manifold_names)
    n_layers = len(true_weights)
    n_cols = n_layers + 1
    n_rows = 2 * n_manifolds

    fate_colors = {
        'surviving': '#2ca02c',  # green
        'flipped': '#d62728',    # red
        'dead': '#888888',       # gray
        'mixed': '#ff7f0e',      # orange
    }

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * n_cols, 3 * n_rows))

    all_data = {}

    for m_idx, name in enumerate(manifold_names):
        kwargs = RELU_MANIFOLD_KWARGS.get(name, {})
        points, metadata = make_manifold(name, n_points=n_points, **kwargs)
        t = np.linspace(0, 1, len(points))
        is_connected = metadata['type'] in ('closed', 'open')
        is_closed = metadata['type'] == 'closed'

        layer_data = propagate_with_relu(points, true_weights, quant_weights)
        all_data[name] = (points, metadata, layer_data)

        data_row = 2 * m_idx
        error_row = 2 * m_idx + 1

        # === DATA ROW: input + transformed manifold at each layer ===
        ax = axes[data_row, 0]
        ax.scatter(points[:, 0], points[:, 1], c=t, cmap='viridis',
                   s=15, edgecolors='none')
        if is_connected:
            conn = np.vstack([points, points[0]]) if is_closed else points
            ax.plot(conn[:, 0], conn[:, 1], 'k-', alpha=0.2, linewidth=0.8)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        ax.axhline(0, color='k', linewidth=0.3)
        ax.axvline(0, color='k', linewidth=0.3)
        if m_idx == 0:
            ax.set_title('Input', fontweight='bold')
        ax.set_ylabel(f'{name}\ndata', fontsize=10, fontweight='bold')
        _square_lims(ax)

        for col, ld in enumerate(layer_data):
            ax = axes[data_row, col + 1]
            transformed = ld['float_post_relu']
            fate = ld['fate']

            # Color data points by fate to match error row
            for fate_name, color in fate_colors.items():
                mask = fate == fate_name
                if mask.any():
                    ax.scatter(transformed[mask, 0], transformed[mask, 1],
                               c=color, s=15, edgecolors='none',
                               zorder=3 if fate_name == 'flipped' else 2)

            if is_connected:
                alive = (fate != 'dead')
                if alive.sum() > 1:
                    alive_pts = transformed[alive]
                    ax.plot(alive_pts[:, 0], alive_pts[:, 1],
                            'k-', alpha=0.15, linewidth=0.5)

            ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
            ax.axhline(0, color='k', linewidth=0.3)
            ax.axvline(0, color='k', linewidth=0.3)
            if m_idx == 0:
                ax.set_title(f'After L{col+1}', fontweight='bold')
            _square_lims(ax)

        # === ERROR ROW: error colored by fate ===
        # Col 0: label
        ax = axes[error_row, 0]
        ax.axis('off')
        ax.text(0.5, 0.5, 'error', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, fontstyle='italic',
                color='#888888')

        for col, ld in enumerate(layer_data):
            ax = axes[error_row, col + 1]
            error = ld['error']
            fate = ld['fate']

            for fate_name, color in fate_colors.items():
                mask = fate == fate_name
                if mask.any():
                    ax.scatter(error[mask, 0], error[mask, 1],
                               c=color, s=15, edgecolors='none',
                               label=fate_name if m_idx == 0 and col == 0 else None,
                               zorder=3 if fate_name == 'flipped' else 2)

            if is_connected:
                alive = (fate != 'dead')
                if alive.sum() > 1:
                    alive_errors = error[alive]
                    ax.plot(alive_errors[:, 0], alive_errors[:, 1],
                            'k-', alpha=0.1, linewidth=0.5)

            ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
            ax.axhline(0, color='k', linewidth=0.3)
            ax.axvline(0, color='k', linewidth=0.3)

            s = ld['stats']
            ax.text(0.95, 0.95,
                    f"S:{s['frac_surviving']:.0%} F:{s['frac_flipped']:.0%}\n"
                    f"D:{s['frac_dead']:.0%} M:{s['frac_mixed']:.0%}",
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=6, family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.8))
            _square_lims(ax)

    # Legend on first error row
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=c, markersize=8, label=n)
               for n, c in fate_colors.items()]
    axes[1, 1].legend(handles=handles, loc='upper left', fontsize=7,
                       framealpha=0.9)

    plt.suptitle('ReLU Interaction with Weight Quantization Error\n'
                 'Data row: W @ x \u2192 ReLU  |  Error row: quant error colored by fate',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('plots/relu_error_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary
    print(f"\n{'Manifold':<15} {'Layer':<7} {'Surviving':<12} {'Flipped':<10} "
          f"{'Dead':<10} {'Mixed':<10}")
    print("-" * 64)
    for name in manifold_names:
        points, metadata, layer_data = all_data[name]
        for i, ld in enumerate(layer_data):
            s = ld['stats']
            print(f"{name if i == 0 else '':<15} L{i+1:<5} "
                  f"{s['frac_surviving']:>8.1%}     {s['frac_flipped']:>6.1%}    "
                  f"{s['frac_dead']:>6.1%}    {s['frac_mixed']:>6.1%}")


def plot_relu_survival_summary(true_weights, manifold_names=None, n_points=128):
    """Summary figure: survival fractions and flip growth across layers.

    Two subplots:
    1. Stacked area chart: surviving/mixed/dead/flipped fractions per layer
    2. Line plot: flip fraction growth across layers for each manifold
    """
    if manifold_names is None:
        manifold_names = ['circle', 'ellipse', 'line', 'spiral', 'figure_eight']

    quant_weights = [quantize(W, DELTA) for W in true_weights]
    n_layers = len(true_weights)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fate_colors_ordered = {
        'surviving': '#2ca02c',
        'mixed': '#ff7f0e',
        'dead': '#888888',
        'flipped': '#d62728',
    }

    # Subplot 1: Stacked area per manifold (one set of stacked bars per manifold)
    ax = axes[0]
    bar_width = 0.15
    x_positions = np.arange(n_layers)

    for m_idx, name in enumerate(manifold_names):
        kwargs = RELU_MANIFOLD_KWARGS.get(name, {})
        points, _ = make_manifold(name, n_points=n_points, **kwargs)
        layer_data = propagate_with_relu(points, true_weights, quant_weights)

        offset = (m_idx - len(manifold_names) / 2 + 0.5) * bar_width
        bottom = np.zeros(n_layers)

        for fate_name, color in fate_colors_ordered.items():
            fracs = np.array([ld['stats'][f'frac_{fate_name}'] for ld in layer_data])
            ax.bar(x_positions + offset, fracs, bar_width,
                   bottom=bottom, color=color, alpha=0.8,
                   edgecolor='white', linewidth=0.3)
            bottom += fracs

        # Label manifold name on top
        ax.text(x_positions[-1] + offset, 1.02, name,
                ha='center', va='bottom', fontsize=6, rotation=45)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'L{i+1}' for i in range(n_layers)])
    ax.set_ylabel('Fraction of points')
    ax.set_title('ReLU Fate Distribution per Layer')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, axis='y')

    # Custom legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.8)
               for c in fate_colors_ordered.values()]
    ax.legend(handles, fate_colors_ordered.keys(), loc='upper left', fontsize=8)

    # Subplot 2: Flip fraction growth
    ax = axes[1]
    manifold_colors = plt.cm.tab10(np.linspace(0, 1, len(manifold_names)))

    for m_idx, (name, color) in enumerate(zip(manifold_names, manifold_colors)):
        kwargs = RELU_MANIFOLD_KWARGS.get(name, {})
        points, _ = make_manifold(name, n_points=n_points, **kwargs)
        layer_data = propagate_with_relu(points, true_weights, quant_weights)

        flip_fracs = [ld['stats']['frac_flipped'] for ld in layer_data]
        ax.plot(range(1, n_layers + 1), flip_fracs, 'o-',
                color=color, linewidth=2, markersize=6, label=name)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Fraction of flipped points')
    ax.set_title('ReLU Flip Fraction Growth\n'
                 '(points where quantization changed ReLU decision)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, n_layers + 1))
    ax.set_xticklabels([f'L{i+1}' for i in range(n_layers)])

    plt.tight_layout()
    plt.savefig('plots/relu_survival_summary.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# Run ReLU analysis
# ============================================================

print("=" * 70)
print("RELU INTERACTION WITH WEIGHT QUANTIZATION ERROR")
print("=" * 70)

relu_weights = [
    np.array([[0.9, 0.2], [0.1, 1.0]]),
    np.array([[0.95, -0.15], [0.2, 0.85]]),
    np.array([[1.0, 0.1], [-0.1, 0.9]]),
    np.array([[0.85, 0.15], [0.1, 1.05]]),
]
relu_qw = [quantize(W, DELTA) for W in relu_weights]

plot_relu_error_evolution(relu_weights, n_points=128)
plot_relu_survival_summary(relu_weights, n_points=128)

print("\nOBSERVATIONS:")
print("1. SURVIVING (green): both paths positive — error is linearly correctable")
print("2. FLIPPED (red): sign disagreement — irreversible ReLU decision change")
print("3. DEAD (gray): both paths negative — ReLU zeros both, error is exactly 0")
print("4. MIXED (orange): partial info loss, surviving part is linearly correctable")
print("5. Flips are rare (~few %) because weight quant error is small vs signal magnitude")
print("6. The geometric 'chopping' by ReLU (dead/mixed) is the dominant effect")

# %%
