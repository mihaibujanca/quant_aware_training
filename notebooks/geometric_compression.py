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
#     display_name: quant-aware-training (3.12.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Geometric Compression: End-to-End Pipeline
#
# Compress a trained overparameterized ReLU MLP (representing an SDF) into a
# much smaller KAN by reading out the geometric structure of the learned function.
#
# Pipeline: Sample SDF → Train source MLP → Geometric profiling → KAN construction → Evaluate

# %% Configuration
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

from aleph.geometric_compression import GeometricCompressionConfig, run_pipeline, save_results
from aleph.mesh_datasets import load_mesh, sample_sdf_points

# --- Config: edit here ---
MESH_NAME = "torus"          # "torus", "sphere", "capsule", "box", "cylinder", or path to .ply/.obj
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
SPEED_TIER = "explore"       # "explore" for iteration, "confirm" for validation, None for full

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

CFG = GeometricCompressionConfig(
    mesh_name=MESH_NAME,
    n_points=100_000,
    surface_ratio=0.5,
    surface_noise=0.01,
    hidden_dim=256,
    depth=4,
    lr=1e-3,
    epochs=500,
    eikonal_lambda=0.1,
    n_clusters=8,
    kan_grid=8,
    jacobian_lambda=1.0,
    kan_fit_steps=500,
    embed_dim=None,   # set to e.g. 16 for high-d embedding experiment
    device=DEVICE,
    seed=SEED,
    speed_tier=SPEED_TIER,
)

RESULT_DIR = Path("runs/geometric_compression") / f"{RUN_ID}_{MESH_NAME}"
PLOT_DIR = RESULT_DIR / "plots"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
print(f"Run ID: {RUN_ID}  →  {RESULT_DIR}")


# --- Visualization helpers ---

def equal_axis_ranges(points=None, limits=None, pad=0.05):
    """Compute equal x/y/z ranges for undistorted 3D plotting."""
    if limits is None:
        points = np.asarray(points)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        limits = np.stack([mins, maxs], axis=1)
    else:
        limits = np.asarray(limits, dtype=float)
    center = limits.mean(axis=1)
    radius = 0.5 * (limits[:, 1] - limits[:, 0]).max() * (1.0 + pad)
    return [[c - radius, c + radius] for c in center]


def build_isosurface_grid(mesh_vertices, grid_res, device, margin=0.25):
    """Build a cubic inference grid that fully covers the mesh with margin."""
    vertices = np.asarray(mesh_vertices)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * (maxs - mins).max() * (1.0 + margin)
    grid_min = center - radius
    grid_max = center + radius
    lin_axes = [
        torch.linspace(float(grid_min[i]), float(grid_max[i]), grid_res, device=device)
        for i in range(3)
    ]
    gx, gy, gz = torch.meshgrid(*lin_axes, indexing='ij')
    grid_pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    spacing = tuple(float((grid_max[i] - grid_min[i]) / (grid_res - 1)) for i in range(3))
    return grid_pts, spacing, grid_min, np.stack([grid_min, grid_max], axis=1)


def plotly_scene_kwargs(ranges):
    """Common scene settings for equal-scale 3D Plotly figures."""
    return dict(
        xaxis=dict(title='x', range=ranges[0], backgroundcolor='rgba(0,0,0,0)'),
        yaxis=dict(title='y', range=ranges[1], backgroundcolor='rgba(0,0,0,0)'),
        zaxis=dict(title='z', range=ranges[2], backgroundcolor='rgba(0,0,0,0)'),
        aspectmode='cube',
        camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
    )


def mesh_trace(verts, faces, color='steelblue', alpha=0.7, name=None):
    """Build a Plotly triangular mesh trace."""
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=alpha, name=name,
        hoverinfo='skip', showlegend=False,
        lighting=dict(ambient=0.5, diffuse=0.75, roughness=0.8, specular=0.15),
    )


def save_plotly_figure(fig, stem):
    out_path = PLOT_DIR / f"{stem}.html"
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f"Saved: {out_path}")


def chamfer_hausdorff_mesh(ref_verts, pred_verts, n_samples=5000):
    from scipy.spatial import cKDTree
    if len(pred_verts) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(0)
    if len(ref_verts) > n_samples:
        ref_verts = ref_verts[rng.choice(len(ref_verts), n_samples, replace=False)]
    if len(pred_verts) > n_samples:
        pred_verts = pred_verts[rng.choice(len(pred_verts), n_samples, replace=False)]
    d_p2r, _ = cKDTree(ref_verts).query(pred_verts)
    d_r2p, _ = cKDTree(pred_verts).query(ref_verts)
    return float(0.5 * (d_p2r.mean() + d_r2p.mean())), float(max(d_p2r.max(), d_r2p.max()))


# %% [markdown]
# ## 1. Single-shape demo: run pipeline and visualize

# %% Run pipeline
print(f"Running pipeline on: {MESH_NAME}")
result = run_pipeline(CFG)
save_results(result, RESULT_DIR)
metrics = result["metrics"]
profile = result["geometric_profile"]

print(f"\nMesh: {profile['mesh_vertices']} vertices, watertight={profile['mesh_watertight']}")
print(f"Source MLP test MSE: {profile['source_test_mse']:.6f}")
print(f"KAN arch: {profile['kan_width']}, grid={profile['kan_grid_used']}")
print(f"Compression: {profile['n_params_source']:,} → {profile['n_params_kan']:,} params "
      f"({profile['n_params_source']/profile['n_params_kan']:.1f}x)")

print("\n" + "=" * 90)
print(f"{'Method':20s} {'Test MSE':>10s} {'Jac MSE':>10s} {'Chamfer':>10s} {'Hausdorff':>11s} {'Params':>9s}")
print("-" * 90)
for name, m in metrics.items():
    ch = m.get('chamfer', float('nan'))
    hd = m.get('hausdorff', float('nan'))
    print(f"{name:20s} {m['test_mse']:10.6f} {m['jacobian_mse']:10.6f} "
          f"{ch:10.5f} {hd:11.5f} {m['params']:9,}")
print("=" * 90)

# %% Visualize: SDF point cloud
mesh = load_mesh(MESH_NAME)
points, sdf = sample_sdf_points(mesh, min(10_000, CFG.n_points), CFG.surface_ratio, CFG.surface_noise)

vis_idx = np.random.choice(len(points), min(5000, len(points)), replace=False)
vis_pts = points[vis_idx].numpy()
vis_sdf = sdf[vis_idx, 0].numpy()

fig = go.Figure(data=[go.Scatter3d(
    x=vis_pts[:, 0], y=vis_pts[:, 1], z=vis_pts[:, 2],
    mode='markers',
    marker=dict(size=2, opacity=0.65, color=vis_sdf, colorscale='RdBu',
                cmin=-0.1, cmax=0.1, colorbar=dict(title='SDF value', len=0.75)),
    hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>SDF=%{marker.color:.4f}<extra></extra>',
    showlegend=False,
)])
fig.update_layout(
    title=f'{MESH_NAME} — SDF point cloud',
    scene=plotly_scene_kwargs(equal_axis_ranges(points=vis_pts)),
    template='plotly_white', width=1000, height=800, margin=dict(l=0, r=0, t=60, b=0),
)
save_plotly_figure(fig, "sdf_point_cloud")
fig.show()

# %% Visualize: curvature heatmap
X_prof_np = result["X_profile"].numpy()   # actual points used for profiling
curv_np = profile["boundary_density"]
curv_log = np.log1p(curv_np)

fig = go.Figure(data=[go.Scatter3d(
    x=X_prof_np[:, 0], y=X_prof_np[:, 1], z=X_prof_np[:, 2],
    mode='markers',
    marker=dict(size=2, opacity=0.7, color=curv_log, colorscale='Hot',
                colorbar=dict(title='log(1 + boundary density)', len=0.75)),
    showlegend=False,
)])
fig.update_layout(
    title=f'{MESH_NAME} — ReLU boundary density',
    scene=plotly_scene_kwargs(equal_axis_ranges(points=X_prof_np)),
    template='plotly_white', margin=dict(l=0, r=0, t=60, b=0),
)
save_plotly_figure(fig, "curvature_heatmap")
fig.show()

# %% Visualize: isosurface comparison (all methods, 2 rows)
from aleph.models import MLP
from aleph.geometric_compression import _make_kan
from skimage.measure import marching_cubes

grid_res = 64
grid_pts, grid_spacing, grid_min, grid_limits = build_isosurface_grid(mesh.vertices, grid_res, DEVICE)
scene_cfg = plotly_scene_kwargs(equal_axis_ranges(limits=grid_limits, pad=0.02))

kan_width = profile["kan_width"]
kan_grid_used = profile["kan_grid_used"]
small_mlp_hidden = profile.get("small_mlp_hidden", 16)
kan_k_used = int(result["config"].get("kan_k", 3))
rbf_width_scale_used = float(result["config"].get("rbf_width_scale", 1.0))

# Re-instantiate models from saved state_dicts (baselines optional)
all_state_dicts = result["all_state_dicts"]

_source = MLP(input_size=profile["input_dim"], hidden_size=CFG.hidden_dim, output_size=1, depth=CFG.depth).to(DEVICE)
_source.load_state_dict(all_state_dicts["source_mlp"]); _source.eval()

_geo_kan = _make_kan(
    kan_width, kan_grid_used, CFG.seed, DEVICE, kan_k=kan_k_used, rbf_width_scale=rbf_width_scale_used
)
_geo_kan.load_state_dict(all_state_dicts["geometric_kan"]); _geo_kan.eval()

if "distilled_kan" in all_state_dicts:
    _dist_kan = _make_kan(
        kan_width, kan_grid_used, CFG.seed, DEVICE, kan_k=kan_k_used, rbf_width_scale=rbf_width_scale_used
    )
    _dist_kan.load_state_dict(all_state_dicts["distilled_kan"]); _dist_kan.eval()
else:
    _dist_kan = None

if "small_mlp" in all_state_dicts and small_mlp_hidden > 0:
    _small = MLP(input_size=profile["input_dim"], hidden_size=small_mlp_hidden, output_size=1, depth=2).to(DEVICE)
    _small.load_state_dict(all_state_dicts["small_mlp"]); _small.eval()
else:
    _small = None

if "kan_scratch" in all_state_dicts:
    _scratch = _make_kan(
        kan_width, kan_grid_used, CFG.seed, DEVICE, kan_k=kan_k_used, rbf_width_scale=rbf_width_scale_used
    )
    _scratch.load_state_dict(all_state_dicts["kan_scratch"]); _scratch.eval()
else:
    _scratch = None


def extract_iso(model, grid_pts, grid_spacing, grid_min, grid_res):
    with torch.no_grad():
        sdf_vol = model(grid_pts).reshape(grid_res, grid_res, grid_res).cpu().numpy()
    try:
        v, f, _, _ = marching_cubes(sdf_vol, level=0.0, spacing=grid_spacing)
        return v + grid_min, f
    except Exception:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)

_KEY_MAP = {"Source MLP": "source_mlp", "Geometric KAN": "geometric_kan",
            "Distilled KAN": "distilled_kan", "Small MLP": "small_mlp", "KAN scratch": "kan_scratch"}

# Layout: row1 = [Original, Source MLP, Geometric KAN]
#         row2 = [Distilled KAN, Small MLP, KAN scratch]
panels = [
    ("Original mesh",   None,       'lightgray'),
    ("Source MLP",      _source,    'steelblue'),
    ("Geometric KAN",   _geo_kan,   'seagreen'),
    ("Distilled KAN",   _dist_kan,  'coral'),
    ("Small MLP",       _small,     'mediumpurple'),
    ("KAN scratch",     _scratch,   'goldenrod'),
]

fig = make_subplots(
    rows=2, cols=3,
    specs=[[{"type": "scene"}]*3, [{"type": "scene"}]*3],
    subplot_titles=[
        title + (f"<br>MSE={metrics[k]['test_mse']:.5f}" if k and k in metrics else
                 ("<br>(skipped)" if k else ""))
        for title, _, __ in panels
        for k in [_KEY_MAP.get(title)]
    ],
    horizontal_spacing=0.02,
    vertical_spacing=0.05,
)

row_col = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
for (title, model, color), (row, col) in zip(panels, row_col):
    if model is None:
        fig.add_trace(mesh_trace(mesh.vertices, mesh.faces, color=color, alpha=0.75, name=title), row=row, col=col)
    else:
        verts, faces = extract_iso(model, grid_pts, grid_spacing, grid_min, grid_res)
        if len(verts):
            fig.add_trace(mesh_trace(verts, faces, color=color, alpha=0.75, name=title), row=row, col=col)
    fig.update_scenes(**scene_cfg, row=row, col=col)

fig.update_layout(
    template='plotly_white',
    margin=dict(l=0, r=0, t=90, b=0),
    height=900,
)
save_plotly_figure(fig, "isosurface_comparison")
fig.show()

# %% [markdown]
# ## 2. Multi-shape sweep
#
# One independent network trained per shape. Results collected for comparison.

# %% Multi-shape sweep
SHAPES = ["sphere", "torus", "capsule", "box", "cylinder"]  # add "meshes/bunny.ply" if available

SWEEP_DIR = Path("runs/geometric_compression") / f"{RUN_ID}_sweep"
SWEEP_PLOT_DIR = SWEEP_DIR / "plots"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_PLOT_DIR.mkdir(exist_ok=True)
print(f"Sweep dir: {SWEEP_DIR}")

SWEEP_CFG = GeometricCompressionConfig(
    n_points=50_000,
    epochs=200,
    kan_fit_steps=200,
    device=DEVICE,
    seed=SEED,
    speed_tier=SPEED_TIER,
)

all_results = {}
for shape in SHAPES:
    print(f"\n{'='*60}")
    print(f"Shape: {shape}")
    print(f"{'='*60}")
    from dataclasses import replace
    cfg = replace(SWEEP_CFG, mesh_name=shape)
    try:
        r = run_pipeline(cfg)
        save_results(r, SWEEP_DIR / shape)
        all_results[shape] = r
        m = r["metrics"]
        p = r["geometric_profile"]
        print(f"  source_mse={p['source_test_mse']:.6f}  "
              f"kan_mse={m['geometric_kan']['test_mse']:.6f}  "
              f"chamfer={m['geometric_kan']['chamfer']:.5f}  "
              f"ratio={p['n_params_source']/p['n_params_kan']:.1f}x")
    except Exception as e:
        print(f"  FAILED: {e}")

# %% Summary table
print("\n" + "=" * 100)
print(f"{'Shape':12s} {'Source MSE':>12s} {'Geom KAN MSE':>14s} {'Chamfer':>10s} "
      f"{'Hausdorff':>12s} {'Ratio':>8s} {'Watertight':>11s}")
print("-" * 100)
for shape, r in all_results.items():
    p = r["geometric_profile"]
    m = r["metrics"]["geometric_kan"]
    print(f"{shape:12s} {p['source_test_mse']:12.6f} {m['test_mse']:14.6f} "
          f"{m['chamfer']:10.5f} {m['hausdorff']:12.5f} "
          f"{p['n_params_source']/p['n_params_kan']:7.1f}x  {str(p['mesh_watertight']):>10s}")
print("=" * 100)

# %% Summary bar charts
shapes_done = list(all_results.keys())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Test MSE comparison
methods_plot = [m for m in ["source_mlp", "geometric_kan", "distilled_kan", "small_mlp", "kan_scratch"]
               if m in next(iter(all_results.values()))["metrics"]]
colors = ['steelblue', 'seagreen', 'coral', 'mediumpurple', 'goldenrod']
x = np.arange(len(shapes_done))
width = 0.15

for j, (method, color) in enumerate(zip(methods_plot, colors)):
    vals = [all_results[s]["metrics"][method]["test_mse"] for s in shapes_done]
    axes[0].bar(x + j * width, vals, width, label=method.replace('_', ' '), color=color, alpha=0.8)
axes[0].set_xticks(x + width * 2)
axes[0].set_xticklabels(shapes_done, rotation=20)
axes[0].set_ylabel("Test MSE")
axes[0].set_title("SDF Test MSE by shape")
axes[0].legend(fontsize=7)
axes[0].set_yscale('log')

# Chamfer distance
for j, (method, color) in enumerate(zip(methods_plot[1:], colors[1:])):
    vals = [all_results[s]["metrics"][method]["chamfer"] for s in shapes_done]
    axes[1].bar(x + j * width, vals, width, label=method.replace('_', ' '), color=color, alpha=0.8)
axes[1].set_xticks(x + width * 1.5)
axes[1].set_xticklabels(shapes_done, rotation=20)
axes[1].set_ylabel("Chamfer distance")
axes[1].set_title("Surface quality (Chamfer) by shape")
axes[1].legend(fontsize=7)

# Compression ratio
ratios = [all_results[s]["geometric_profile"]["n_params_source"] /
          all_results[s]["geometric_profile"]["n_params_kan"]
          for s in shapes_done]
axes[2].bar(shapes_done, ratios, color='steelblue', alpha=0.8)
axes[2].set_ylabel("Compression ratio (source/KAN)")
axes[2].set_title("Parameter compression ratio")
plt.setp(axes[2].get_xticklabels(), rotation=20)

plt.tight_layout()
plt.savefig(SWEEP_PLOT_DIR / "multi_shape_comparison.png", dpi=150)
plt.show()
print(f"Saved: {SWEEP_PLOT_DIR / 'multi_shape_comparison.png'}")

# %%
