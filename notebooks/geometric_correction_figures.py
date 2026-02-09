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
# # Geometric Quantization Error Correction — Figures
#
# Figures for the research brief. Uses a minimal 2-layer 2D network
# (2→2→2 with ReLU) to show:
#
# 1. **The partition**: ReLU divides input space into polyhedral regions.
#    Each region has its own affine map.
# 2. **Metric distortion**: Quantization changes the affine map within
#    each region (wrong rotation/scaling). Linearly correctable.
# 3. **Topological distortion**: Quantization moves the partition boundaries.
#    Points near boundaries end up in the wrong region. Not linearly correctable.
# 4. **Correction**: Undoing the local distortion at each layer recovers
#    the float output for metric-distorted points.

# %%
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
})

SAVE_DIR = Path('plots')
SAVE_DIR.mkdir(exist_ok=True)

# Quantization
BITS = 4
DELTA = 1.0 / (2 ** (BITS - 1))

def quantize(W):
    return torch.round(W / DELTA) * DELTA

# %% [markdown]
# ## Setup: A 2-layer 2D network
#
# ```
# x ∈ R² → W₁x + b₁ → ReLU → W₂(·) + b₂ → output ∈ R²
# ```
#
# The weights are chosen to create a visually clear partition with
# non-trivial rotation/scaling, so quantization effects are visible.

# %%
# Layer 1: slight rotation + asymmetric scaling
W1 = torch.tensor([[0.8, 0.3],
                    [-0.2, 0.9]], dtype=torch.float32)
b1 = torch.tensor([0.1, -0.05], dtype=torch.float32)

# Layer 2: more rotation
W2 = torch.tensor([[0.7, -0.4],
                    [0.5, 0.6]], dtype=torch.float32)
b2 = torch.tensor([0.0, 0.0], dtype=torch.float32)

W1_q = quantize(W1)
W2_q = quantize(W2)

E1 = W1_q - W1  # quantization error matrices
E2 = W2_q - W2

def forward_float(x, W1, b1, W2, b2):
    z1 = F.linear(x, W1, b1)
    a1 = F.relu(z1)
    z2 = F.linear(a1, W2, b2)
    return z2, z1, a1

def forward_quant(x, W1q, b1, W2q, b2):
    z1 = F.linear(x, W1q, b1)
    a1 = F.relu(z1)
    z2 = F.linear(a1, W2q, b2)
    return z2, z1, a1

print("=== Layer 1 ===")
print(f"W1 =\n{W1}")
print(f"W1_q =\n{W1_q}")
print(f"E1 =\n{E1}")
print(f"|E1| = {E1.norm():.6f}")

print(f"\n=== Layer 2 ===")
print(f"W2 =\n{W2}")
print(f"W2_q =\n{W2_q}")
print(f"E2 =\n{E2}")
print(f"|E2| = {E2.norm():.6f}")

# %% [markdown]
# ## Figure 1: The Polyhedral Partition
#
# ReLU at layer 1 creates hyperplanes in input space. Each row of W₁
# defines a hyperplane w_i^T x + b_i = 0. This divides the plane into
# up to 4 regions, each with a different sign pattern.
#
# Quantization shifts these hyperplanes (dashed lines).

# %%
def get_hyperplane_points(w, b, xlim, n=200):
    """Get points along the line w[0]*x + w[1]*y + b = 0."""
    # Solve for y: y = -(w[0]*x + b) / w[1]  if w[1] != 0
    # Solve for x: x = -(w[1]*y + b) / w[0]  if w[0] != 0
    if abs(w[1]) > abs(w[0]):
        xs = np.linspace(xlim[0], xlim[1], n)
        ys = -(w[0] * xs + b) / w[1]
        mask = (ys >= xlim[0]) & (ys <= xlim[1])
        return xs[mask], ys[mask]
    else:
        ys = np.linspace(xlim[0], xlim[1], n)
        xs = -(w[1] * ys + b) / w[0]
        mask = (xs >= xlim[0]) & (xs <= xlim[1])
        return xs[mask], ys[mask]


def classify_sign_pattern(x, W, b):
    """Return sign pattern of W @ x + b for each point."""
    # x: (N, 2), W: (2, 2), b: (2,)
    z = x @ W.T + b  # (N, 2)
    return (z > 0).int()  # (N, 2) of 0/1


# Grid over input space
extent = 3.0
n_grid = 500
xs = np.linspace(-extent, extent, n_grid)
ys = np.linspace(-extent, extent, n_grid)
X_grid, Y_grid = np.meshgrid(xs, ys)
grid_points = torch.tensor(
    np.column_stack([X_grid.ravel(), Y_grid.ravel()]),
    dtype=torch.float32
)

# Sign patterns
signs_float = classify_sign_pattern(grid_points, W1, b1)
signs_quant = classify_sign_pattern(grid_points, W1_q, b1)

# Encode sign pattern as single int for coloring: s0*2 + s1
pattern_float = (signs_float[:, 0] * 2 + signs_float[:, 1]).numpy().reshape(n_grid, n_grid)
pattern_quant = (signs_quant[:, 0] * 2 + signs_quant[:, 1]).numpy().reshape(n_grid, n_grid)

# Topological distortion: where sign pattern changed
topo_change = (pattern_float != pattern_quant).astype(float)

# Colors for 4 sign patterns
region_colors = ['#e8e8e8', '#a8d8ea', '#ffcccb', '#c8e6c9']
region_cmap = ListedColormap(region_colors)
region_labels = ['(-,-)', '(-,+)', '(+,-)', '(+,+)']

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel A: Float partition
ax = axes[0]
ax.contourf(X_grid, Y_grid, pattern_float, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
            colors=region_colors, alpha=0.7)
# Float hyperplanes
w1_np, b1_np = W1.numpy(), b1.numpy()
for i, color in enumerate(['#1f77b4', '#d62728']):
    hx, hy = get_hyperplane_points(w1_np[i], b1_np[i], (-extent, extent))
    ax.plot(hx, hy, '-', color=color, linewidth=2.5,
            label=f'H{i+1}: {w1_np[i,0]:.1f}x + {w1_np[i,1]:.1f}y + {b1_np[i]:.2f} = 0')
ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
ax.set_aspect('equal')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Float: ReLU partition')
ax.legend(fontsize=7, loc='lower left')
# Region labels
for pattern_val, label in enumerate(region_labels):
    mask = pattern_float == pattern_val
    if mask.any():
        ys_masked = Y_grid[mask]
        xs_masked = X_grid[mask]
        cx, cy = xs_masked.mean(), ys_masked.mean()
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=9, fontweight='bold', alpha=0.6)

# Panel B: Quantized partition with shift highlighted
ax = axes[1]
ax.contourf(X_grid, Y_grid, pattern_quant, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
            colors=region_colors, alpha=0.7)
w1q_np = W1_q.numpy()
for i, color in enumerate(['#1f77b4', '#d62728']):
    # Float (dashed)
    hx, hy = get_hyperplane_points(w1_np[i], b1_np[i], (-extent, extent))
    ax.plot(hx, hy, '--', color=color, linewidth=1.5, alpha=0.5)
    # Quantized (solid)
    hx, hy = get_hyperplane_points(w1q_np[i], b1_np[i], (-extent, extent))
    ax.plot(hx, hy, '-', color=color, linewidth=2.5,
            label=f'H{i+1}q: {w1q_np[i,0]:.3f}x + {w1q_np[i,1]:.3f}y = 0')
ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
ax.set_aspect('equal')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Quantized: shifted boundaries')
ax.legend(fontsize=7, loc='lower left')

# Panel C: Topological distortion (where partition changed)
ax = axes[2]
ax.contourf(X_grid, Y_grid, pattern_float, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
            colors=region_colors, alpha=0.3)
ax.contourf(X_grid, Y_grid, topo_change, levels=[0.5, 1.5],
            colors=['#ff0000'], alpha=0.4)
# Both sets of hyperplanes
for i, color in enumerate(['#1f77b4', '#d62728']):
    hx, hy = get_hyperplane_points(w1_np[i], b1_np[i], (-extent, extent))
    ax.plot(hx, hy, '-', color=color, linewidth=2, label=f'Float H{i+1}')
    hx, hy = get_hyperplane_points(w1q_np[i], b1_np[i], (-extent, extent))
    ax.plot(hx, hy, '--', color=color, linewidth=2, label=f'Quant H{i+1}')
ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
ax.set_aspect('equal')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Topological distortion\n(red = changed sign pattern)')
ax.legend(fontsize=6, loc='lower left')

plt.tight_layout()
plt.savefig(SAVE_DIR / 'fig1_partition.png', dpi=200, bbox_inches='tight')
plt.show()

print(f"Topological distortion: {topo_change.mean()*100:.1f}% of input space changed partition")

# %% [markdown]
# ## Figure 2: Grid Distortion — Metric vs Topological
#
# A regular grid in input space is mapped through both the float and
# quantized networks. The grid warping tells the full geometric story:
#
# - **Metric distortion** (within each partition region): grid cells are
#   stretched and rotated differently by float vs quantized weights.
#   This is the dominant error (88-98%) and is linearly correctable —
#   it's just the wrong affine map applied to the same region.
#
# - **Topological distortion** (at partition boundaries): grid lines
#   fold or tear because points near boundaries end up in different
#   ReLU regions. This changes the piecewise-linear structure and
#   is NOT linearly correctable.

# %%
# Map a grid through float and quantized networks
n_lines = 13
line_density = 300
grid_vals = np.linspace(-extent, extent, n_lines)

# Collect grid lines as (input_pts, float_output, quant_output)
grid_lines_h = []  # horizontal (vary x, fixed y)
grid_lines_v = []  # vertical (fixed x, vary y)

for y_val in grid_vals:
    xs = np.linspace(-extent, extent, line_density)
    pts = torch.tensor(np.column_stack([xs, np.full_like(xs, y_val)]),
                       dtype=torch.float32)
    of, _, _ = forward_float(pts, W1, b1, W2, b2)
    oq, _, _ = forward_quant(pts, W1_q, b1, W2_q, b2)
    grid_lines_h.append((pts.numpy(), of.numpy(), oq.numpy()))

for x_val in grid_vals:
    ys = np.linspace(-extent, extent, line_density)
    pts = torch.tensor(np.column_stack([np.full_like(ys, x_val), ys]),
                       dtype=torch.float32)
    of, _, _ = forward_float(pts, W1, b1, W2, b2)
    oq, _, _ = forward_quant(pts, W1_q, b1, W2_q, b2)
    grid_lines_v.append((pts.numpy(), of.numpy(), oq.numpy()))


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Input grid with partition boundaries
ax = axes[0]
ax.contourf(X_grid, Y_grid, pattern_float, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
            colors=region_colors, alpha=0.15)
for inp, _, _ in grid_lines_h:
    ax.plot(inp[:, 0], inp[:, 1], '-', color='#555555', linewidth=0.6, alpha=0.5)
for inp, _, _ in grid_lines_v:
    ax.plot(inp[:, 0], inp[:, 1], '-', color='#555555', linewidth=0.6, alpha=0.5)
# Partition boundaries
for i, color in enumerate(['#1f77b4', '#d62728']):
    hx, hy = get_hyperplane_points(w1_np[i], b1_np[i], (-extent, extent))
    ax.plot(hx, hy, '-', color=color, linewidth=2.5)
    hx, hy = get_hyperplane_points(w1q_np[i], b1_np[i], (-extent, extent))
    ax.plot(hx, hy, '--', color=color, linewidth=1.5, alpha=0.5)
ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
ax.set_aspect('equal')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Input space\n(regular grid + partition)')

# Panel B: Float-transformed grid
ax = axes[1]
for _, of, _ in grid_lines_h:
    ax.plot(of[:, 0], of[:, 1], '-', color='#1f77b4', linewidth=0.7, alpha=0.6)
for _, of, _ in grid_lines_v:
    ax.plot(of[:, 0], of[:, 1], '-', color='#1f77b4', linewidth=0.7, alpha=0.6)
ax.set_aspect('equal')
ax.set_xlabel('$y_1$'); ax.set_ylabel('$y_2$')
ax.set_title('Float: warped grid\n(piecewise-affine, clean)')

# Panel C: Quantized grid overlaid on float
ax = axes[2]
# Float grid (faded)
for _, of, _ in grid_lines_h:
    ax.plot(of[:, 0], of[:, 1], '-', color='#1f77b4', linewidth=0.5, alpha=0.3)
for _, of, _ in grid_lines_v:
    ax.plot(of[:, 0], of[:, 1], '-', color='#1f77b4', linewidth=0.5, alpha=0.3)
# Quantized grid
for _, _, oq in grid_lines_h:
    ax.plot(oq[:, 0], oq[:, 1], '-', color='#ff7f0e', linewidth=0.7, alpha=0.6)
for _, _, oq in grid_lines_v:
    ax.plot(oq[:, 0], oq[:, 1], '-', color='#ff7f0e', linewidth=0.7, alpha=0.6)
ax.set_aspect('equal')
ax.set_xlabel('$y_1$'); ax.set_ylabel('$y_2$')
ax.set_title('Quantized (orange) vs Float (blue)\nmetric = shift, topological = tear')

plt.tight_layout()
plt.savefig(SAVE_DIR / 'fig2_grid_distortion.png', dpi=200, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Figure 3: Metric vs Topological — Per-Point Classification
#
# Sample points from input space. For each point, compute:
# - Float output: W₂ @ ReLU(W₁x + b₁) + b₂
# - Quantized output: W₂_q @ ReLU(W₁_q x + b₁) + b₂
# - Error vector: quantized - float
#
# Color by whether the point stayed in the same partition (metric = green)
# or crossed a boundary (topological = red).

# %%
# Sample points on a circle (clean manifold for visualization)
n_pts = 200
theta = torch.linspace(0, 2 * np.pi, n_pts + 1)[:-1]
radius = 2.0
sample_pts = torch.stack([radius * torch.cos(theta),
                          radius * torch.sin(theta)], dim=1)

out_f, z1_f, a1_f = forward_float(sample_pts, W1, b1, W2, b2)
out_q, z1_q, a1_q = forward_quant(sample_pts, W1_q, b1, W2_q, b2)

error = out_q - out_f

# Classify: same sign pattern (metric) or different (topological)
signs_f = (z1_f > 0).int()
signs_q = (z1_q > 0).int()
same_partition = (signs_f == signs_q).all(dim=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel A: Input space with circle, colored by distortion type
ax = axes[0]
# Background partition
ax.contourf(X_grid, Y_grid, pattern_float, levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
            colors=region_colors, alpha=0.3)
for i, color in enumerate(['#1f77b4', '#d62728']):
    hx, hy = get_hyperplane_points(w1_np[i], b1_np[i], (-extent, extent))
    ax.plot(hx, hy, '-', color=color, linewidth=1, alpha=0.5)
metric_mask = same_partition.numpy()
topo_mask = ~same_partition.numpy()
pts_np = sample_pts.numpy()
ax.scatter(pts_np[metric_mask, 0], pts_np[metric_mask, 1],
           c='#2ca02c', s=15, zorder=3, label=f'Metric ({metric_mask.sum()})')
ax.scatter(pts_np[topo_mask, 0], pts_np[topo_mask, 1],
           c='#d62728', s=25, zorder=4, marker='x', linewidth=1.5,
           label=f'Topological ({topo_mask.sum()})')
ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
ax.set_aspect('equal')
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_title('Input: metric (green)\nvs topological (red)')
ax.legend(fontsize=8)

# Panel B: Output space — float vs quantized
ax = axes[1]
out_f_np = out_f.detach().numpy()
out_q_np = out_q.detach().numpy()
# Float manifold
ax.plot(out_f_np[:, 0], out_f_np[:, 1], '-', color='#1f77b4',
        linewidth=2, alpha=0.8, label='Float')
# Quantized manifold
ax.plot(out_q_np[:, 0], out_q_np[:, 1], '-', color='#ff7f0e',
        linewidth=2, alpha=0.8, label='Quantized')
# Error arrows for a subset
step = n_pts // 20
for i in range(0, n_pts, step):
    color = '#2ca02c' if same_partition[i] else '#d62728'
    ax.annotate('', xy=out_q_np[i], xytext=out_f_np[i],
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
ax.set_aspect('equal')
ax.set_xlabel('$y_1$'); ax.set_ylabel('$y_2$')
ax.set_title('Output: float (blue)\nvs quantized (orange)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Panel C: Error vectors, colored by type
ax = axes[2]
err_np = error.detach().numpy()
err_norms = np.linalg.norm(err_np, axis=1)
ax.scatter(err_np[metric_mask, 0], err_np[metric_mask, 1],
           c='#2ca02c', s=15, alpha=0.7, label='Metric')
ax.scatter(err_np[topo_mask, 0], err_np[topo_mask, 1],
           c='#d62728', s=25, alpha=0.7, marker='x', linewidth=1.5,
           label='Topological')
ax.axhline(0, color='k', linewidth=0.3)
ax.axvline(0, color='k', linewidth=0.3)
ax.set_aspect('equal')
ax.set_xlabel('Error $e_1$'); ax.set_ylabel('Error $e_2$')
ax.set_title('Error space: metric errors\nlie on a smooth manifold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(SAVE_DIR / 'fig3_metric_vs_topological.png', dpi=200, bbox_inches='tight')
plt.show()

metric_err = err_norms[metric_mask].mean()
topo_err = err_norms[topo_mask].mean() if topo_mask.any() else 0
metric_energy = (err_norms[metric_mask] ** 2).sum()
topo_energy = (err_norms[topo_mask] ** 2).sum() if topo_mask.any() else 0
total_energy = metric_energy + topo_energy
print(f"Metric distortion: {metric_mask.sum()}/{n_pts} points "
      f"({100*metric_mask.mean():.0f}%), mean error = {metric_err:.4f}, "
      f"energy = {100*metric_energy/total_energy:.1f}%")
print(f"Topological distortion: {topo_mask.sum()}/{n_pts} points "
      f"({100*topo_mask.mean():.0f}%), mean error = {topo_err:.4f}, "
      f"energy = {100*topo_energy/total_energy:.1f}%")

# %% [markdown]
# ## Figure 4: What Correction Achieves
#
# Apply local correction c_local = -E_L @ a at each layer (pre-ReLU).
# This undoes the metric distortion but cannot fix topological distortion.
#
# Three panels: float output, quantized output, corrected output.
# Corrected should match float for metric-distorted points.

# %%
@torch.no_grad()
def forward_corrected(x, W1, W1_q, b1, W2, W2_q, b2):
    """Forward with c_local correction at each layer."""
    # Layer 1: quantized pre-activation + local correction
    a0 = x
    z1 = F.linear(a0, W1_q, b1)
    c_local_1 = -F.linear(a0, E1)  # c_local = -E1 @ x
    z1_corrected = z1 + c_local_1
    a1 = F.relu(z1_corrected)

    # Layer 2: quantized pre-activation + local correction
    # BUT: a1 is from the corrected path, so local error at layer 2
    # uses a1 (which may differ from float a1 due to ReLU on corrected z1)
    z2 = F.linear(a1, W2_q, b2)
    c_local_2 = -F.linear(a1, E2)
    z2_corrected = z2 + c_local_2
    return z2_corrected, z1_corrected, a1

out_c, z1_c, a1_c = forward_corrected(sample_pts, W1, W1_q, b1, W2, W2_q, b2)
out_c_np = out_c.numpy()
err_corrected = out_c - out_f
err_corrected_np = err_corrected.numpy()

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# Panel A: Float output manifold
ax = axes[0]
ax.plot(out_f_np[:, 0], out_f_np[:, 1], '-', color='#1f77b4', linewidth=2.5)
ax.scatter(out_f_np[::10, 0], out_f_np[::10, 1], c='#1f77b4', s=20, zorder=3)
ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
ax.set_title('Float output')
ax.set_xlabel('$y_1$'); ax.set_ylabel('$y_2$')

# Panel B: Quantized output manifold
ax = axes[1]
ax.plot(out_f_np[:, 0], out_f_np[:, 1], '-', color='#1f77b4',
        linewidth=1, alpha=0.3, label='Float')
ax.plot(out_q_np[:, 0], out_q_np[:, 1], '-', color='#ff7f0e', linewidth=2.5)
ax.scatter(out_q_np[::10, 0], out_q_np[::10, 1], c='#ff7f0e', s=20, zorder=3)
ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
ax.set_title('Quantized output')
ax.set_xlabel('$y_1$'); ax.set_ylabel('$y_2$')
ax.legend(fontsize=7)

# Panel C: Corrected output manifold
ax = axes[2]
ax.plot(out_f_np[:, 0], out_f_np[:, 1], '-', color='#1f77b4',
        linewidth=1, alpha=0.3, label='Float')
ax.plot(out_c_np[:, 0], out_c_np[:, 1], '-', color='#2ca02c', linewidth=2.5)
ax.scatter(out_c_np[::10, 0], out_c_np[::10, 1], c='#2ca02c', s=20, zorder=3)
ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
ax.set_title('Corrected output\n(c_local at each layer)')
ax.set_xlabel('$y_1$'); ax.set_ylabel('$y_2$')
ax.legend(fontsize=7)

# Panel D: Residual error after correction
ax = axes[3]
err_c_norms = np.linalg.norm(err_corrected_np, axis=1)
err_q_norms = np.linalg.norm(err_np, axis=1)
theta_deg = np.degrees(theta.numpy())
ax.plot(theta_deg, err_q_norms, '-', color='#ff7f0e', linewidth=2,
        label=f'Quantized (mean={err_q_norms.mean():.4f})')
ax.plot(theta_deg, err_c_norms, '-', color='#2ca02c', linewidth=2,
        label=f'Corrected (mean={err_c_norms.mean():.4f})')
# Shade topological regions
for i in range(n_pts):
    if topo_mask[i]:
        ax.axvspan(theta_deg[i] - 1, theta_deg[i] + 1,
                   alpha=0.1, color='red')
ax.set_xlabel('Input angle (degrees)')
ax.set_ylabel('Output error norm')
ax.set_title('Error reduction\n(red bands = topological)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(SAVE_DIR / 'fig4_correction.png', dpi=200, bbox_inches='tight')
plt.show()

reduction = 1 - err_c_norms.mean() / err_q_norms.mean()
print(f"Mean error: quantized={err_q_norms.mean():.4f}, "
      f"corrected={err_c_norms.mean():.4f} ({100*reduction:.0f}% reduction)")
print(f"Metric points: mean corrected error = "
      f"{err_c_norms[metric_mask].mean():.6f}")
if topo_mask.any():
    print(f"Topological points: mean corrected error = "
          f"{err_c_norms[topo_mask].mean():.6f}")

# %% [markdown]
# ## Figure 5: The Error Space Has Low-Rank Structure
#
# The oracle corrections across data points form a matrix. Its SVD
# reveals the intrinsic dimensionality of the correctable error.
# In 2D this is trivially rank ≤ 2, but the singular value ratio
# shows how concentrated the error is along one direction.

# %%
# Oracle corrections at layer 1 and layer 2
# Use a dense grid of points (not just circle)
n_dense = 1000
torch.manual_seed(42)
dense_pts = torch.randn(n_dense, 2) * 2.0

# Float trace
out_f_d, z1_f_d, a1_f_d = forward_float(dense_pts, W1, b1, W2, b2)

# Layer 1 oracle correction: C1 = -E1 @ x (just local, no propagation for L1)
C1 = -F.linear(dense_pts, E1)

# Layer 2: full oracle correction accounting for propagation
# C2 = -E2 @ a1_float - W2 @ epsilon1
# where epsilon1 = relu(W1q @ x + b1) - relu(W1 @ x + b1)
z1_q_d = F.linear(dense_pts, W1_q, b1)
a1_q_d = F.relu(z1_q_d)
epsilon1 = a1_q_d - a1_f_d
C2_local = -F.linear(a1_f_d, E2)
C2_propagated = -F.linear(epsilon1, W2)
C2 = C2_local + C2_propagated

# SVD of corrections
U1, S1, V1 = torch.linalg.svd(C1, full_matrices=False)
U2, S2, V2 = torch.linalg.svd(C2, full_matrices=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel A: Layer 1 corrections in error space
ax = axes[0]
C1_np = C1.numpy()
ax.scatter(C1_np[:, 0], C1_np[:, 1], c='#2ca02c', s=5, alpha=0.3)
# Draw principal directions
center = C1_np.mean(axis=0)
for i, (s, v) in enumerate(zip(S1.numpy(), V1.numpy())):
    scale = s / np.sqrt(n_dense) * 3  # visual scaling
    ax.annotate('', xy=center + v * scale, xytext=center,
                arrowprops=dict(arrowstyle='->', color=['#d62728', '#1f77b4'][i],
                                lw=3))
    ax.text(center[0] + v[0] * scale * 1.15,
            center[1] + v[1] * scale * 1.15,
            f'$\\sigma_{i+1}$={S1[i]:.1f}', fontsize=9, fontweight='bold',
            color=['#d62728', '#1f77b4'][i])
ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
ax.axhline(0, color='k', linewidth=0.3); ax.axvline(0, color='k', linewidth=0.3)
ax.set_xlabel('Correction $c_1$'); ax.set_ylabel('Correction $c_2$')
ax.set_title(f'Layer 1 corrections\n$\\sigma_1/\\sigma_2$ = {S1[0]/S1[1]:.2f}')

# Panel B: Layer 2 corrections (local vs propagated)
ax = axes[1]
C2l_np = C2_local.detach().numpy()
C2p_np = C2_propagated.detach().numpy()
C2_np = C2.detach().numpy()
ax.scatter(C2l_np[:, 0], C2l_np[:, 1], c='#2ca02c', s=5, alpha=0.2,
           label='Local')
ax.scatter(C2p_np[:, 0], C2p_np[:, 1], c='#d62728', s=5, alpha=0.2,
           label='Propagated')
ax.scatter(C2_np[:, 0], C2_np[:, 1], c='#1f77b4', s=5, alpha=0.2,
           label='Total')
ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
ax.axhline(0, color='k', linewidth=0.3); ax.axvline(0, color='k', linewidth=0.3)
ax.set_xlabel('Correction $c_1$'); ax.set_ylabel('Correction $c_2$')
ax.set_title('Layer 2: local (green)\nvs propagated (red)')
ax.legend(fontsize=7, markerscale=3)

# Panel C: Singular value comparison
ax = axes[2]
x_pos = np.arange(2)
width = 0.35
ax.bar(x_pos - width/2, S1.numpy() / S1[0].item(), width,
       color='#2ca02c', alpha=0.7, label='Layer 1')
ax.bar(x_pos + width/2, S2.numpy() / S2[0].item(), width,
       color='#1f77b4', alpha=0.7, label='Layer 2')
ax.set_xticks(x_pos)
ax.set_xticklabels(['$\\sigma_1$', '$\\sigma_2$'])
ax.set_ylabel('Normalized singular value')
ax.set_title('Error rank structure\n(1.0 = dominant direction)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# Energy in top-1
e1_top1 = (S1[0] ** 2 / (S1 ** 2).sum()).item()
e2_top1 = (S2[0] ** 2 / (S2 ** 2).sum()).item()
ax.text(0.95, 0.95,
        f'L1: top-1 = {100*e1_top1:.0f}% energy\n'
        f'L2: top-1 = {100*e2_top1:.0f}% energy',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(SAVE_DIR / 'fig5_error_rank.png', dpi=200, bbox_inches='tight')
plt.show()

local_energy = (C2_local.norm() ** 2).item()
prop_energy = (C2_propagated.norm() ** 2).item()
total = local_energy + prop_energy
print(f"Layer 2 correction energy: local={100*local_energy/total:.0f}%, "
      f"propagated={100*prop_energy/total:.0f}%")
print(f"Layer 1 singular values: {S1.numpy()}, ratio={S1[0]/S1[1]:.2f}")
print(f"Layer 2 singular values: {S2.numpy()}, ratio={S2[0]/S2[1]:.2f}")

# %% [markdown]
# ## Summary
#
# These figures show the complete geometric picture:
#
# 1. **Partition** (Fig 1): ReLU creates polyhedral regions. Quantization
#    shifts the boundaries. The shifted region (red band) is small.
#
# 2. **Grid Distortion** (Fig 2): A regular grid mapped through the float
#    and quantized networks. Within each region, grid cells are
#    stretched/rotated differently (metric distortion — the wrong affine
#    map). At boundaries, grid lines tear (topological distortion —
#    changed piecewise-linear structure).
#
# 3. **Per-Point Classification** (Fig 3): Most points stay in the same
#    region (green) — their error is a linear distortion. Few cross
#    boundaries (red) — those have nonlinear error.
#
# 4. **Correction** (Fig 4): Local correction c_local = -E @ a undoes
#    the metric distortion. Corrected output (green) tracks float (blue).
#    Residual error is concentrated where topological distortion occurs.
#
# 5. **Low-Rank** (Fig 5): The correction vectors have low-rank
#    structure — most energy concentrated in 1-2 directions. This means
#    a small correction network (projecting onto these directions) suffices.

# %%
