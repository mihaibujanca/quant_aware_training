"""Geometric compression pipeline: MLP → KAN distillation on 3D mesh SDFs.

Reusable pipeline for training a source MLP on a mesh SDF, profiling its
geometric structure, and compressing into a KAN using Jacobian matching.
"""

from __future__ import annotations

import json
import math
import warnings

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from skimage.measure import marching_cubes

from aleph.differential import gradient as compute_gradient, relu_boundary_density
from aleph.mesh_datasets import load_mesh, sample_sdf_points, make_embedding_matrix
from aleph.models import MLP


# ---------------------------------------------------------------------------
# Projected model wrapper
# ---------------------------------------------------------------------------

class ProjectedModel(nn.Module):
    """Wraps a model with a fixed linear projection applied to inputs.

    forward(x) = model(x @ projection)
    projection: [input_dim, k]
    """

    def __init__(self, projection: torch.Tensor, model: nn.Module):
        super().__init__()
        self.projection = nn.Parameter(projection, requires_grad=False)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x @ self.projection)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_METHOD_NAMES = ("source_mlp", "geometric_kan", "projected_kan", "distilled_kan", "small_mlp", "kan_scratch")
_DEFAULT_METHODS = ("source_mlp", "geometric_kan", "small_mlp")
_DEFAULT_SURFACE_METRIC_METHODS = _METHOD_NAMES


@dataclass
class GeometricCompressionConfig:
    # Data
    mesh_name: str = "torus"
    n_points: int = 100_000
    surface_ratio: float = 0.5
    surface_noise: float = 0.01

    # Source MLP
    hidden_dim: int = 256
    depth: int = 4
    lr: float = 1e-3
    epochs: int = 500
    eikonal_lambda: float = 0.1

    # KAN
    n_clusters: int = 8
    kan_grid: int = 8  # num_grids (RBF centers) for FastKAN
    kan_k: int = 3     # compatibility proxy for FastKAN smoothness
    rbf_width_scale: float = 1.0
    jacobian_lambda: float = 1.0
    jacobian_schedule: str = "constant"  # "constant", "warmup", "cooldown"
    kan_fit_steps: int = 500

    # Embedding (None = native 3D)
    embed_dim: Optional[int] = None
    embed_seed: int = 42

    # Infra
    device: str = "cpu"
    seed: int = 42
    batch_size: int = 4096
    n_profile: int = 10_000
    n_fit: int = 5_000
    grid_res: int = 64
    jacobian_eval_points: int = 1_000

    # Speed tier
    speed_tier: Optional[str] = None   # "explore", "confirm", or None (use explicit values)
    n_perturbations: int = 32
    kmeans_n_init: int = 10
    lbfgs_max_iter: int = 20
    skip_baselines: bool = False
    methods: tuple[str, ...] = _DEFAULT_METHODS
    compute_surface_metrics: bool = True
    surface_metric_methods: tuple[str, ...] = _DEFAULT_SURFACE_METRIC_METHODS
    eikonal_every_n_steps: int = 1
    eikonal_subsample: Optional[int] = None
    projection_rank: int = 3


# ---------------------------------------------------------------------------
# Speed tier system
# ---------------------------------------------------------------------------

_SPEED_TIERS = {
    "explore": dict(
        epochs=260,
        kan_fit_steps=130,
        n_points=45_000,
        n_profile=4_500,
        n_fit=2_800,
        grid_res=64,
        n_perturbations=14,
        kmeans_n_init=8,
        lbfgs_max_iter=14,
        skip_baselines=False,
        methods=_DEFAULT_METHODS,
        compute_surface_metrics=True,
        surface_metric_methods=_DEFAULT_SURFACE_METRIC_METHODS,
        eikonal_every_n_steps=1,
        eikonal_subsample=None,
    ),
    "confirm": dict(
        epochs=260,
        kan_fit_steps=130,
        n_points=45_000,
        n_profile=4_500,
        n_fit=2_800,
        grid_res=64,
        n_perturbations=14,
        kmeans_n_init=8,
        lbfgs_max_iter=14,
        skip_baselines=False,
        methods=_METHOD_NAMES,
        compute_surface_metrics=True,
        surface_metric_methods=_DEFAULT_SURFACE_METRIC_METHODS,
        eikonal_every_n_steps=1,
        eikonal_subsample=None,
    ),
}


def apply_speed_tier(cfg: GeometricCompressionConfig) -> GeometricCompressionConfig:
    """Apply speed-tier defaults to fields still at dataclass defaults."""
    if cfg.speed_tier is None:
        return cfg
    overrides = _SPEED_TIERS.get(cfg.speed_tier)
    if overrides is None:
        raise ValueError(f"Unknown speed_tier {cfg.speed_tier!r}. Choose 'explore', 'confirm', or None.")
    defaults = GeometricCompressionConfig()
    effective = {
        k: v
        for k, v in overrides.items()
        if getattr(cfg, k) == getattr(defaults, k)
    }
    return replace(cfg, **effective)


def _normalize_methods(cfg: GeometricCompressionConfig) -> tuple[str, ...]:
    methods = tuple(dict.fromkeys(cfg.methods))
    invalid = [m for m in methods if m not in _METHOD_NAMES]
    if invalid:
        raise ValueError(f"Unknown method(s) in cfg.methods: {invalid}. Valid: {_METHOD_NAMES}.")

    if cfg.skip_baselines:
        warnings.warn(
            "`skip_baselines` is deprecated; prefer cfg.methods.",
            DeprecationWarning,
            stacklevel=2,
        )
        methods = tuple(m for m in methods if m in ("source_mlp", "geometric_kan"))

    # Source/geometric KAN are required for pipeline semantics.
    if "source_mlp" not in methods:
        methods = ("source_mlp",) + methods
    if "geometric_kan" not in methods:
        methods = tuple(m for m in methods if m != "source_mlp")
        methods = ("source_mlp", "geometric_kan") + methods

    return methods


def _normalize_surface_methods(
    surface_methods: tuple[str, ...],
    methods: tuple[str, ...],
) -> tuple[str, ...]:
    if not surface_methods:
        return methods

    invalid = [m for m in surface_methods if m not in _METHOD_NAMES]
    if invalid:
        raise ValueError(
            f"Unknown method(s) in cfg.surface_metric_methods: {invalid}. Valid: {_METHOD_NAMES}."
        )
    return tuple(m for m in methods if m in set(surface_methods))


def _kan_k_to_rbf_width_scale(kan_k: int) -> float:
    """Compatibility map from legacy spline order k to FastKAN RBF width scale."""
    raw = float(kan_k) / 3.0
    return float(min(4.0, max(0.25, raw)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mlp_param_count(input_dim: int, h: int, depth: int) -> int:
    return input_dim * h + h + sum(h * h + h for _ in range(depth - 1)) + h + 1


def _make_kan(
    kan_width,
    num_grids,
    seed,
    device,
    kan_k: int = 3,
    rbf_width_scale: float = 1.0,
):
    from fastkan import FastKAN
    torch.manual_seed(seed)
    model = FastKAN(layers_hidden=kan_width, num_grids=num_grids).to(device)

    # FastKAN has no spline-order parameter. Preserve legacy kan_k intent by
    # scaling RBF basis width: larger => smoother/wider basis, smaller => sharper.
    width_scale = _kan_k_to_rbf_width_scale(kan_k) * float(rbf_width_scale)
    if not math.isclose(width_scale, 1.0, rel_tol=1e-6, abs_tol=1e-8):
        for layer in getattr(model, "layers", []):
            if hasattr(layer, "rbf") and hasattr(layer.rbf, "denominator"):
                layer.rbf.denominator *= width_scale
    return model


def _fit_kan_lbfgs(kan_model, X_fit, y_fit, grad_fit, jacobian_lambda, steps, device,
                   max_iter=20, patience=10, jacobian_schedule="constant"):
    """Fit KAN via L-BFGS with Jacobian-matching loss. Returns loss list.

    Args:
        max_iter: L-BFGS max_iter per step (controls closure evaluations).
        patience: stop early if loss hasn't improved by >0.1% for this many steps.
        jacobian_schedule: "constant" (default), "warmup" (0→lambda), "cooldown" (lambda→0).
    """
    optimizer = torch.optim.LBFGS(
        kan_model.parameters(), lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe",
    )
    losses = []
    best_loss = float("inf")
    no_improve = 0
    for step_idx in range(steps):
        # Compute step-dependent lambda
        t = step_idx / max(steps - 1, 1)  # 0 → 1
        if jacobian_schedule == "warmup":
            lam = jacobian_lambda * t
        elif jacobian_schedule == "cooldown":
            lam = jacobian_lambda * (1.0 - t)
        else:
            lam = jacobian_lambda

        def closure(lam=lam):
            optimizer.zero_grad()
            x_in = X_fit.detach().requires_grad_(True)
            pred = kan_model(x_in)
            out_loss = F.mse_loss(pred, y_fit)
            grad_kan = torch.autograd.grad(
                pred.sum(), x_in, create_graph=True, retain_graph=True
            )[0]
            loss = out_loss + lam * F.mse_loss(grad_kan, grad_fit)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        loss_val = loss.item()
        losses.append(loss_val)
        if loss_val < best_loss * (1.0 - 1e-3):
            best_loss = loss_val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return losses


def _train_kan_adam(kan_model, X_fit, y_fit, epochs, batch_size, device):
    """Train KAN via Adam (distillation baseline)."""
    optimizer = torch.optim.Adam(kan_model.parameters(), lr=1e-3)
    n = len(X_fit)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = kan_model(X_fit[idx])
            loss = F.mse_loss(pred, y_fit[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _train_mlp_adam(model, X_train, y_train, epochs, batch_size, device):
    """Train MLP via Adam (no eikonal)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(X_train)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X_train[idx])
            loss = F.mse_loss(pred, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()


def _build_isosurface_grid(mesh_vertices, grid_res, device, margin=0.25):
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
    return grid_pts, spacing, grid_min


def _chamfer_hausdorff(ref_verts, pred_verts, n_samples=5000):
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


def _reconstruct_mesh(model, grid_pts_3d, A, grid_spacing, grid_min, grid_res, device):
    """Evaluate model on grid and extract isosurface via marching cubes.

    Always builds the grid in 3D, embeds if A is not None.
    Returns (verts, faces) or (empty, empty) on failure.
    """
    grid_query = grid_pts_3d @ A.T if A is not None else grid_pts_3d
    with torch.no_grad():
        grid_sdf = model(grid_query).reshape(grid_res, grid_res, grid_res).cpu().numpy()
    try:
        verts, faces, _, _ = marching_cubes(grid_sdf, level=0.0, spacing=grid_spacing)
        verts += grid_min
        return verts, faces
    except Exception:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)


def jacobian_rank_profile(model, X, chunk_size=512):
    """SVD profile of gradient ensemble G=[N, d] for scalar-output models.

    For scalar-output models, each gradient is a vector in R^d. Stacking
    gradients over many input points yields a matrix whose singular spectrum
    reveals effective dimensionality of the model's differential behavior.

    Args:
        model: scalar-output model f: R^d -> R
        X: [N, d] input points
        chunk_size: gradient chunk size

    Returns:
        Dict with singular values, energy ratios, cumulative energy, and
        effective rank summaries.
    """
    G = compute_gradient(model, X, chunk_size=chunk_size)  # [N, d]
    if G.dim() != 2:
        raise ValueError(f"Expected gradient matrix [N, d], got shape {tuple(G.shape)}")
    if G.shape[0] < 2:
        raise ValueError("Need at least 2 samples for rank profiling.")

    # Use full SVD of gradient ensemble, as specified in phase-2 analysis.
    singular_values = torch.linalg.svdvals(G)  # [min(N, d)], sorted desc
    energy = singular_values.square()
    total_energy = energy.sum()

    if total_energy <= 0:
        energy_ratio = torch.zeros_like(energy)
        cumulative_energy = torch.zeros_like(energy)
        effective_rank = 0.0
    else:
        energy_ratio = energy / total_energy
        cumulative_energy = torch.cumsum(energy_ratio, dim=0)
        p = energy_ratio.clamp_min(1e-12)
        effective_rank = float(torch.exp(-(p * p.log()).sum()).item())

    def _rank_at(threshold: float) -> int:
        if len(cumulative_energy) == 0:
            return 0
        hit = torch.nonzero(cumulative_energy >= threshold, as_tuple=False)
        if len(hit) == 0:
            return int(len(cumulative_energy))
        return int(hit[0].item() + 1)

    return {
        "singular_values": singular_values.detach().cpu(),
        "energy_ratio": energy_ratio.detach().cpu(),
        "cumulative_energy": cumulative_energy.detach().cpu(),
        "effective_rank": effective_rank,
        "rank_90": _rank_at(0.90),
        "rank_95": _rank_at(0.95),
        "rank_99": _rank_at(0.99),
        "n_samples": int(G.shape[0]),
        "input_dim": int(G.shape[1]),
    }


def jacobian_projection(model: nn.Module, X: torch.Tensor, k: int, chunk_size: int = 512) -> torch.Tensor:
    """Extract top-k right singular vectors of the Jacobian as a projection matrix.

    Args:
        model: scalar-output model f: R^d -> R
        X: [N, d] input points
        k: number of projection dimensions
        chunk_size: gradient chunk size

    Returns:
        P: [d, k] projection matrix such that X_proj = X @ P has shape [N, k]
    """
    G = compute_gradient(model, X, chunk_size=chunk_size)  # [N, d]
    _, _, Vh = torch.linalg.svd(G, full_matrices=False)    # Vh: [min(N,d), d]
    return Vh[:k].T  # [d, k]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: GeometricCompressionConfig) -> dict:
    """Run the full geometric compression pipeline for one shape.

    Returns a dict with keys:
        config, metrics, source_state_dict, kan_state_dict, geometric_profile
    """
    cfg = apply_speed_tier(cfg)
    methods = _normalize_methods(cfg)
    surface_metric_methods = _normalize_surface_methods(cfg.surface_metric_methods, methods)
    cfg = replace(cfg, methods=methods, surface_metric_methods=surface_metric_methods)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device

    # ------------------------------------------------------------------
    # 1. Load mesh and sample SDF
    # ------------------------------------------------------------------
    mesh = load_mesh(cfg.mesh_name)
    points, sdf = sample_sdf_points(mesh, cfg.n_points, cfg.surface_ratio, cfg.surface_noise)

    # Embedding: embed 3D points into higher-dimensional space
    A = None
    input_dim = 3
    if cfg.embed_dim is not None:
        A = make_embedding_matrix(3, cfg.embed_dim, seed=cfg.embed_seed).to(device)
        input_dim = cfg.embed_dim

    # Split train/test in 3D; embedding applied on-device
    n_train = int(0.8 * len(points))
    perm = torch.randperm(len(points))
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    X_train_3d = points[train_idx].to(device)
    X_test_3d = points[test_idx].to(device)
    y_train = sdf[train_idx].to(device)
    y_test = sdf[test_idx].to(device)

    # Apply embedding
    if A is not None:
        X_train = X_train_3d @ A.T  # [N, embed_dim]
        X_test = X_test_3d @ A.T
    else:
        X_train, X_test = X_train_3d, X_test_3d

    # ------------------------------------------------------------------
    # 2. Train source MLP with eikonal regularization
    # ------------------------------------------------------------------
    source_model = MLP(
        input_size=input_dim,
        hidden_size=cfg.hidden_dim,
        output_size=1,
        depth=cfg.depth,
    ).to(device)

    optimizer = torch.optim.Adam(source_model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    eikonal_every = max(1, int(cfg.eikonal_every_n_steps))
    global_step = 0
    for epoch in range(cfg.epochs):
        source_model.train()
        perm_e = torch.randperm(len(X_train), device=device)
        for i in range(0, len(X_train), cfg.batch_size):
            idx = perm_e[i:i + cfg.batch_size]

            do_eikonal = cfg.eikonal_lambda > 0 and (global_step % eikonal_every == 0)
            if do_eikonal and cfg.eikonal_subsample is None:
                x_batch = X_train[idx].requires_grad_(True)
                pred = source_model(x_batch)
                mse_loss = F.mse_loss(pred, y_train[idx])
                grad_out = torch.autograd.grad(
                    pred.sum(), x_batch, create_graph=True, retain_graph=True
                )[0]
                eikonal_loss = ((grad_out.norm(dim=-1) - 1.0) ** 2).mean()
            else:
                x_batch = X_train[idx]
                pred = source_model(x_batch)
                mse_loss = F.mse_loss(pred, y_train[idx])
                eikonal_loss = pred.new_tensor(0.0)
                if do_eikonal:
                    sub_n = min(int(cfg.eikonal_subsample), len(idx))
                    if sub_n > 0:
                        if sub_n == len(idx):
                            idx_sub = idx
                        else:
                            sub_sel = torch.randperm(len(idx), device=device)[:sub_n]
                            idx_sub = idx[sub_sel]
                        x_eik = X_train[idx_sub].detach().requires_grad_(True)
                        pred_eik = source_model(x_eik)
                        grad_out = torch.autograd.grad(
                            pred_eik.sum(), x_eik, create_graph=True, retain_graph=True
                        )[0]
                        eikonal_loss = ((grad_out.norm(dim=-1) - 1.0) ** 2).mean()

            loss = mse_loss + cfg.eikonal_lambda * eikonal_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
        scheduler.step()

    source_model.eval()
    with torch.no_grad():
        source_test_mse = F.mse_loss(source_model(X_test), y_test).item()

    # ------------------------------------------------------------------
    # 3. Geometric profiling
    # ------------------------------------------------------------------
    n_profile = min(cfg.n_profile, len(X_train))
    X_profile = X_train[:n_profile]

    J = compute_gradient(source_model, X_profile, chunk_size=512)  # [N, input_dim]
    J_np = J.cpu().numpy()

    curv = relu_boundary_density(source_model, X_profile, eps=0.01, n_perturbations=cfg.n_perturbations, chunk_size=512)
    curv_np = curv.cpu().numpy()
    p50, p90 = float(np.percentile(curv_np, 50)), float(np.percentile(curv_np, 90))

    # Determine KAN grid (auto-scale for high curvature)
    kan_grid = cfg.kan_grid
    curvature_ratio = p90 / max(p50, 1e-6)
    if curvature_ratio > 5:
        kan_grid = min(cfg.kan_grid * 2, 20)

    # Jacobian clustering → KAN width
    km = KMeans(n_clusters=cfg.n_clusters, n_init=cfg.kmeans_n_init, random_state=cfg.seed).fit(J_np)
    kan_width = [input_dim, cfg.n_clusters, 1]

    # ------------------------------------------------------------------
    # 4. Fitting subset
    # ------------------------------------------------------------------
    n_fit = min(cfg.n_fit, len(X_train))
    X_fit = X_train[:n_fit].detach()
    with torch.no_grad():
        y_fit = source_model(X_fit)
    grad_fit = compute_gradient(source_model, X_fit, chunk_size=512).detach()

    # ------------------------------------------------------------------
    # 5. Geometric KAN (Jacobian matching, L-BFGS)
    # ------------------------------------------------------------------
    kan_model = _make_kan(
        kan_width,
        kan_grid,
        cfg.seed,
        device,
        kan_k=cfg.kan_k,
        rbf_width_scale=cfg.rbf_width_scale,
    )
    _fit_kan_lbfgs(kan_model, X_fit, y_fit, grad_fit, cfg.jacobian_lambda, cfg.kan_fit_steps, device,
                   max_iter=cfg.lbfgs_max_iter, jacobian_schedule=cfg.jacobian_schedule)

    # ------------------------------------------------------------------
    # 5b. Projected KAN (low-rank projection from Jacobian SVD)
    # ------------------------------------------------------------------
    # J is already computed above: [n_profile, input_dim]
    requested = set(cfg.methods)
    projected_kan_model = None
    if "projected_kan" in requested:
        k = cfg.projection_rank
        _, _, Vh = torch.linalg.svd(J.detach(), full_matrices=False)
        P = Vh[:k].T  # [input_dim, k]

        X_fit_proj = X_fit @ P        # [n_fit, k]
        grad_fit_proj = grad_fit @ P  # [n_fit, k]

        proj_kan_width = [k, cfg.n_clusters, 1]
        inner_kan = _make_kan(
            proj_kan_width,
            kan_grid,
            cfg.seed,
            device,
            kan_k=cfg.kan_k,
            rbf_width_scale=cfg.rbf_width_scale,
        )
        _fit_kan_lbfgs(inner_kan, X_fit_proj, y_fit, grad_fit_proj, cfg.jacobian_lambda,
                       cfg.kan_fit_steps, device, max_iter=cfg.lbfgs_max_iter,
                       jacobian_schedule=cfg.jacobian_schedule)
        projected_kan_model = ProjectedModel(P, inner_kan)

    # ------------------------------------------------------------------
    # 6. Baselines
    # ------------------------------------------------------------------
    need_distill = "distilled_kan" in requested
    need_small = "small_mlp" in requested
    need_scratch = "kan_scratch" in requested

    n_params_kan = sum(p.numel() for p in kan_model.parameters())
    best_h = 0

    if need_distill:
        kan_distill = _make_kan(
            kan_width,
            kan_grid,
            cfg.seed,
            device,
            kan_k=cfg.kan_k,
            rbf_width_scale=cfg.rbf_width_scale,
        )
        _train_kan_adam(kan_distill, X_fit, y_fit, cfg.epochs, cfg.batch_size, device)
    else:
        kan_distill = None

    if need_small:
        best_h = 4
        for h in range(4, 256):
            if _mlp_param_count(input_dim, h, 2) > n_params_kan:
                best_h = h - 1
                break
        small_mlp = MLP(input_size=input_dim, hidden_size=best_h, output_size=1, depth=2).to(device)
        _train_mlp_adam(small_mlp, X_train, y_train, cfg.epochs, cfg.batch_size, device)
    else:
        small_mlp = None

    if need_scratch:
        kan_scratch = _make_kan(
            kan_width,
            kan_grid,
            cfg.seed,
            device,
            kan_k=cfg.kan_k,
            rbf_width_scale=cfg.rbf_width_scale,
        )
        _train_kan_adam(kan_scratch, X_train, y_train, cfg.epochs, cfg.batch_size, device)
    else:
        kan_scratch = None

    # ------------------------------------------------------------------
    # 7. Evaluate all models
    # ------------------------------------------------------------------
    all_models = {}
    if "source_mlp" in requested:
        all_models["source_mlp"] = source_model
    if "geometric_kan" in requested:
        all_models["geometric_kan"] = kan_model
    if projected_kan_model is not None:
        all_models["projected_kan"] = projected_kan_model
    if need_distill and kan_distill is not None:
        all_models["distilled_kan"] = kan_distill
    if need_small and small_mlp is not None:
        all_models["small_mlp"] = small_mlp
    if need_scratch and kan_scratch is not None:
        all_models["kan_scratch"] = kan_scratch

    n_params_source = sum(p.numel() for p in source_model.parameters())

    metrics = {}
    n_jac_eval = min(cfg.jacobian_eval_points, len(X_test))
    X_jac_eval = X_test[:n_jac_eval]
    G_source = compute_gradient(source_model, X_jac_eval, chunk_size=512)
    for name, model in all_models.items():
        model.eval()
        with torch.no_grad():
            pred = model(X_test)
            mse = F.mse_loss(pred, y_test).item()
        G_model = G_source if name == "source_mlp" else compute_gradient(model, X_jac_eval, chunk_size=512)
        jac_mse = F.mse_loss(G_model, G_source).item()
        n_params = sum(p.numel() for p in model.parameters())
        metrics[name] = {
            "test_mse": mse,
            "jacobian_mse": jac_mse,
            "params": n_params,
            "compression_ratio": n_params_source / n_params,
        }

    # ------------------------------------------------------------------
    # 8. Mesh reconstruction and surface metrics
    # ------------------------------------------------------------------
    if cfg.compute_surface_metrics and cfg.surface_metric_methods:
        grid_pts_3d, grid_spacing, grid_min = _build_isosurface_grid(
            mesh.vertices, cfg.grid_res, device
        )
        ref_verts = mesh.vertices

        for name in cfg.surface_metric_methods:
            model = all_models.get(name)
            if model is None:
                continue
            verts, _ = _reconstruct_mesh(model, grid_pts_3d, A, grid_spacing, grid_min, cfg.grid_res, device)
            ch, hd = _chamfer_hausdorff(ref_verts, verts)
            metrics[name]["chamfer"] = ch
            metrics[name]["hausdorff"] = hd

    # ------------------------------------------------------------------
    # 9. Assemble result
    # ------------------------------------------------------------------
    geometric_profile = {
        "jacobian_cluster_labels": km.labels_,
        "jacobian_cluster_centers": km.cluster_centers_,
        "boundary_density": curv_np,
        "boundary_density_p50": p50,
        "boundary_density_p90": p90,
        "curvature_ratio": float(curvature_ratio),
        "kan_grid_used": kan_grid,
        "kan_width": kan_width,
        "n_params_source": n_params_source,
        "n_params_kan": n_params_kan,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_watertight": bool(mesh.is_watertight),
        "source_test_mse": source_test_mse,
        "input_dim": input_dim,
        "small_mlp_hidden": int(best_h),
    }

    all_state_dicts = {
        "source_mlp": {k: v.cpu() for k, v in source_model.state_dict().items()},
        "geometric_kan": {k: v.cpu() for k, v in kan_model.state_dict().items()},
    }
    if projected_kan_model is not None:
        all_state_dicts["projected_kan"] = {k: v.cpu() for k, v in projected_kan_model.state_dict().items()}
    if kan_distill is not None:
        all_state_dicts["distilled_kan"] = {k: v.cpu() for k, v in kan_distill.state_dict().items()}
    if small_mlp is not None:
        all_state_dicts["small_mlp"] = {k: v.cpu() for k, v in small_mlp.state_dict().items()}
    if kan_scratch is not None:
        all_state_dicts["kan_scratch"] = {k: v.cpu() for k, v in kan_scratch.state_dict().items()}

    return {
        "config": asdict(cfg),
        "metrics": metrics,
        "source_state_dict": all_state_dicts["source_mlp"],   # keep for backwards compat
        "kan_state_dict": all_state_dicts["geometric_kan"],
        "all_state_dicts": all_state_dicts,
        "geometric_profile": geometric_profile,
        "X_profile": X_profile.cpu(),   # [n_profile, input_dim] — for viz only
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: dict, output_dir: Path) -> None:
    """Save pipeline results to output_dir.

    Creates:
        config.json          — all config values
        metrics.json         — per-method metrics
        source_model.pt      — source MLP state_dict
        kan_model.pt         — geometric KAN state_dict
        geometric_profile.pt — numpy/tensor profile data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(results["config"], f, indent=2)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results["metrics"], f, indent=2)

    torch.save(results["source_state_dict"], output_dir / "source_model.pt")
    torch.save(results["kan_state_dict"], output_dir / "kan_model.pt")

    # Save all model state_dicts
    if "all_state_dicts" in results:
        for name, sd in results["all_state_dicts"].items():
            torch.save(sd, output_dir / f"{name}_model.pt")

    # geometric_profile contains numpy arrays — convert for torch.save
    profile = results["geometric_profile"]
    profile_tensors = {}
    for k, v in profile.items():
        if isinstance(v, np.ndarray):
            profile_tensors[k] = torch.from_numpy(v)
        else:
            profile_tensors[k] = v
    torch.save(profile_tensors, output_dir / "geometric_profile.pt")
