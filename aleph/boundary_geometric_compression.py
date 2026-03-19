"""Curvature-adaptive boundary compression for embedded ReLU classifiers."""

from __future__ import annotations

import copy
import json
import math
import time

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.path import Path
from scipy.interpolate import CubicSpline, griddata
from scipy.stats import spearmanr
from skimage.measure import find_contours
from sklearn.neighbors import NearestNeighbors

from aleph.datasets import (
    EmbeddedClassificationBenchmark,
    make_embedded_boundary_classification_benchmark,
)
from aleph.differential import gradient, jacobian_variation
from aleph.models import MLP
from aleph.relu_geometric_compression import (
    _device_for,
    _eval_classifier,
    _forward_hidden,
    _make_classifier,
    _mlp_param_count,
    _num_params,
    _serialize,
    _set_seed,
    _targets,
    _tensor,
)


@dataclass
class BoundaryCompressionConfig:
    datasets: tuple[str, ...] = ("windowed_ripple_ring", "serpentine_tube", "corrugated_dumbbell")
    observed_dims: tuple[int, ...] = (2, 100, 1000)
    gate_observed_dim: int = 100
    n_samples: int = 2400
    noise: float = 0.08
    boundary_fraction: float = 0.55
    n_boundary_points: int = 1024
    val_fraction: float = 0.2
    test_fraction: float = 0.2

    source_hidden: int = 64
    source_depth: int = 4
    source_epochs: int = 1000
    source_lr: float = 1e-3

    small_hidden: int = 32
    small_depth: int = 4
    baseline_epochs: int = 1000
    baseline_lr: float = 5e-4
    tiny_depth_options: tuple[int, ...] = (1, 2, 3)

    projection_rank: int = 2
    projection_method: str = "manifold_pca"
    projected_knn_k: int = 12
    projected_curvature_eps: float = 0.02
    projected_boundary_eps: float = 0.04
    projected_boundary_perturbations: int = 8
    high_complexity_quantile: float = 0.65

    grid_size: int = 192
    grid_margin: float = 0.15
    chunk_min_points: int = 16
    low_chunk_tolerance: float = 0.03
    low_chunk_tolerance_growth: float = 1.35
    low_chunk_max_tolerance: float = 0.18
    arc_fit_gain: float = 0.75

    min_spline_knots: int = 4
    max_spline_knots: int = 32
    chunk_sample_points: int = 80
    polygon_sample_density: int = 18

    target_compression_ratio: int = 10
    fallback_compression_ratio: int = 5
    target_trainable_cap: int = 640
    fallback_trainable_cap: int = 1280
    min_trainable_budget: int = 96

    gate_accuracy_margin: float = 0.05
    gate_bce_ratio: float = 0.80

    calibration_steps: int = 250
    calibration_lr: float = 5e-2
    trace_every_n_steps: int = 0

    device: str = "auto"
    seed: int = 42


@dataclass(frozen=True)
class TeacherBoundaryProfile:
    projection_method: str
    projection_P: np.ndarray
    projection_mean: np.ndarray
    projected_train: np.ndarray
    projected_val: np.ndarray
    projected_test: np.ndarray
    projected_logits_grid: np.ndarray
    projected_complexity_field: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    teacher_boundary_curve: np.ndarray
    teacher_boundary_complexity: np.ndarray
    teacher_boundary_true_curvature: np.ndarray
    boundary_chunk_labels: np.ndarray
    complexity_correlation: float
    projection_subspace_alignment: float


@dataclass(frozen=True)
class BoundaryChunk:
    chunk_id: int
    kind: str
    is_high_complexity: bool
    mean_complexity: float
    mean_true_curvature: float
    polyline_points: np.ndarray | None = None
    arc_center: np.ndarray | None = None
    arc_radius: float | None = None
    arc_start_angle: float | None = None
    arc_end_angle: float | None = None
    frame_origin: np.ndarray | None = None
    frame_tangent: np.ndarray | None = None
    spline_u_knots: np.ndarray | None = None
    spline_v_knots: np.ndarray | None = None
    sampled_curve: np.ndarray | None = None
    trainable_params: int = 0
    stored_numbers: int = 0


@dataclass(frozen=True)
class HybridBoundaryModel:
    projection_P: np.ndarray
    projection_mean: np.ndarray
    low_curvature_chunks: tuple[BoundaryChunk, ...]
    high_curvature_chunks: tuple[BoundaryChunk, ...]
    boundary_polygon: np.ndarray
    temperature: float
    bias: float
    trainable_params: int
    stored_numbers: int
    target_budget: int
    selected_budget_name: str


class _ProjectedTeacherView(nn.Module):
    def __init__(self, teacher: nn.Module, projection_P: np.ndarray, projection_mean: np.ndarray):
        super().__init__()
        self.teacher = teacher
        self.register_buffer("P", torch.tensor(projection_P, dtype=torch.float32))
        self.register_buffer("mean", torch.tensor(projection_mean, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.mean + z @ self.P.T
        return self.teacher(x)


class _ConstantLogit(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(len(x), 1)


def _normalize_scores(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, 5))
    hi = float(np.percentile(x, 95))
    if hi <= lo + 1e-8:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _subspace_alignment_score(embedding_W: np.ndarray, projection_P: np.ndarray) -> float:
    q_true, _ = np.linalg.qr(embedding_W.T)
    q_proj, _ = np.linalg.qr(projection_P)
    svals = np.linalg.svd(q_true.T @ q_proj, compute_uv=False)
    return float(np.mean(svals[: min(len(svals), projection_P.shape[1])]))


def _project_points(X_high: np.ndarray, projection_P: np.ndarray, projection_mean: np.ndarray) -> np.ndarray:
    return ((X_high - projection_mean) @ projection_P).astype(np.float32)


def _train_classifier_with_best_checkpoint(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor | None = None,
    y_test: torch.Tensor | None = None,
    trace_every_n_steps: int = 0,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trace: list[dict[str, float]] = []
    best_val_bce = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None

    for step in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        loss.backward()
        optimizer.step()

        val_metrics = _eval_classifier(model, X_val, y_val)
        if val_metrics["bce"] < best_val_bce:
            best_val_bce = float(val_metrics["bce"])
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "step": float(step + 1),
                "val_bce": float(val_metrics["bce"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            }
            if X_test is not None and y_test is not None:
                test_metrics = _eval_classifier(model, X_test, y_test)
                best_metrics["test_bce"] = float(test_metrics["bce"])
                best_metrics["test_accuracy"] = float(test_metrics["accuracy"])

        if trace_every_n_steps > 0 and ((step + 1) % trace_every_n_steps == 0 or step == 0 or step + 1 == epochs):
            row = {
                "step": float(step + 1),
                "train_loss": float(loss.detach().item()),
                "val_bce": float(val_metrics["bce"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            }
            if X_test is not None and y_test is not None:
                test_metrics = _eval_classifier(model, X_test, y_test)
                row["test_bce"] = float(test_metrics["bce"])
                row["test_accuracy"] = float(test_metrics["accuracy"])
            trace.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    return trace, best_metrics or {}


def _collect_model_metrics(
    model: nn.Module,
    benchmark: EmbeddedClassificationBenchmark,
    *,
    device: torch.device,
) -> dict[str, float]:
    X_val = _tensor(benchmark.val.X_high, device)
    y_val = _targets(benchmark.val.y, device)
    X_test = _tensor(benchmark.test.X_high, device)
    y_test = _targets(benchmark.test.y, device)
    val_metrics = _eval_classifier(model, X_val, y_val)
    test_metrics = _eval_classifier(model, X_test, y_test)
    return {
        "params": float(_num_params(model)),
        "val_bce": float(val_metrics["bce"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "test_bce": float(test_metrics["bce"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_margin": float(test_metrics["mean_margin"]),
    }


def extract_teacher_projection(
    teacher: nn.Module,
    X_high: np.ndarray,
    *,
    projection_rank: int,
    projection_method: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    projection_mean = X_high.mean(axis=0).astype(np.float32)
    centered = (X_high - projection_mean).astype(np.float32)
    if projection_method == "manifold_pca":
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    elif projection_method == "jacobian_svd":
        x = _tensor(X_high, device)
        grads = gradient(teacher, x, chunk_size=256).detach().cpu().numpy().astype(np.float32)
        _, _, vt = np.linalg.svd(grads, full_matrices=False)
    else:
        raise ValueError(
            f"Unsupported projection_method {projection_method!r}. "
            "Choose 'manifold_pca' or 'jacobian_svd'."
        )
    basis = vt[:projection_rank].T.astype(np.float32)
    q, _ = np.linalg.qr(basis)
    projection_P = q[:, :projection_rank].astype(np.float32)
    return projection_P, projection_mean


def _projected_boundary_density(
    teacher: nn.Module,
    projected_points: np.ndarray,
    projection_P: np.ndarray,
    projection_mean: np.ndarray,
    *,
    eps: float,
    n_perturbations: int,
    device: torch.device,
) -> np.ndarray:
    z = torch.tensor(projected_points, dtype=torch.float32, device=device)
    P = torch.tensor(projection_P, dtype=torch.float32, device=device)
    mean = torch.tensor(projection_mean, dtype=torch.float32, device=device)
    density = torch.zeros(len(z), dtype=torch.float32, device=device)

    for _ in range(n_perturbations):
        direction = torch.randn_like(z)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
        z_pert = z + eps * direction
        x_base = mean + z @ P.T
        x_pert = mean + z_pert @ P.T
        _, base_masks, _, _ = _forward_hidden(teacher, x_base)
        _, pert_masks, _, _ = _forward_hidden(teacher, x_pert)
        base = torch.cat([m.float() for m in base_masks], dim=1)
        pert = torch.cat([m.float() for m in pert_masks], dim=1)
        density += (base != pert).float().sum(dim=1)

    return (density / max(n_perturbations, 1)).detach().cpu().numpy().astype(np.float32)


def _activation_churn(
    teacher: nn.Module,
    X_high: np.ndarray,
    projected_points: np.ndarray,
    *,
    knn_k: int,
    device: torch.device,
) -> np.ndarray:
    x = _tensor(X_high, device)
    _, masks, _, _ = _forward_hidden(teacher, x)
    codes = torch.cat([m.float() for m in masks], dim=1).detach().cpu().numpy()
    if len(projected_points) <= 1 or codes.shape[1] == 0:
        return np.zeros(len(projected_points), dtype=np.float32)
    k = min(knn_k + 1, len(projected_points))
    nbrs = NearestNeighbors(n_neighbors=k).fit(projected_points)
    neigh = nbrs.kneighbors(projected_points, return_distance=False)[:, 1:]
    churn = np.abs(codes[:, None, :] - codes[neigh]).mean(axis=2).mean(axis=1)
    return churn.astype(np.float32)


def _make_projected_grid(
    all_points: np.ndarray,
    *,
    grid_size: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    mins = mins - margin * span
    maxs = maxs + margin * span
    grid_x = np.linspace(mins[0], maxs[0], grid_size, dtype=np.float32)
    grid_y = np.linspace(mins[1], maxs[1], grid_size, dtype=np.float32)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    points = np.column_stack([mesh_x.reshape(-1), mesh_y.reshape(-1)]).astype(np.float32)
    return grid_x, grid_y, points


def _extract_longest_contour(field: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    contours = find_contours(field, level=0.0)
    if not contours:
        raise RuntimeError("Could not extract a zero contour from the projected teacher grid.")
    contour = max(contours, key=len)
    row_idx = contour[:, 0]
    col_idx = contour[:, 1]
    x = np.interp(col_idx, np.arange(len(grid_x)), grid_x)
    y = np.interp(row_idx, np.arange(len(grid_y)), grid_y)
    pts = np.column_stack([x, y]).astype(np.float32)
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.vstack([pts, pts[:1]])
    return pts


def _interpolate_field(
    src_points: np.ndarray,
    src_values: np.ndarray,
    query_points: np.ndarray,
) -> np.ndarray:
    linear = griddata(src_points, src_values, query_points, method="linear")
    nearest = griddata(src_points, src_values, query_points, method="nearest")
    if linear is None:
        return nearest.astype(np.float32)
    out = np.where(np.isnan(linear), nearest, linear)
    return out.astype(np.float32)


def build_teacher_boundary_profile(
    teacher: nn.Module,
    benchmark: EmbeddedClassificationBenchmark,
    cfg: BoundaryCompressionConfig,
    *,
    device: torch.device,
) -> TeacherBoundaryProfile:
    projection_P, projection_mean = extract_teacher_projection(
        teacher,
        benchmark.train.X_high,
        projection_rank=cfg.projection_rank,
        projection_method=cfg.projection_method,
        device=device,
    )

    z_train = _project_points(benchmark.train.X_high, projection_P, projection_mean)
    z_val = _project_points(benchmark.val.X_high, projection_P, projection_mean)
    z_test = _project_points(benchmark.test.X_high, projection_P, projection_mean)
    z_all = np.vstack([z_train, z_val, z_test]).astype(np.float32)

    projected_teacher = _ProjectedTeacherView(teacher, projection_P, projection_mean).to(device)
    z_train_t = torch.tensor(z_train, dtype=torch.float32, device=device)
    churn = _activation_churn(teacher, benchmark.train.X_high, z_train, knn_k=cfg.projected_knn_k, device=device)
    jac_var = jacobian_variation(
        projected_teacher,
        z_train_t,
        eps=cfg.projected_curvature_eps,
        chunk_size=256,
    ).detach().cpu().numpy().astype(np.float32)
    boundary_density = _projected_boundary_density(
        teacher,
        z_train,
        projection_P,
        projection_mean,
        eps=cfg.projected_boundary_eps,
        n_perturbations=cfg.projected_boundary_perturbations,
        device=device,
    )
    complexity_score = (
        _normalize_scores(churn)
        + _normalize_scores(jac_var)
        + _normalize_scores(boundary_density)
    ) / 3.0

    grid_x, grid_y, grid_points = _make_projected_grid(z_all, grid_size=cfg.grid_size, margin=cfg.grid_margin)
    with torch.no_grad():
        logits_grid = projected_teacher(_tensor(grid_points, device)).detach().cpu().numpy().reshape(
            cfg.grid_size, cfg.grid_size
        )
    complexity_grid = _interpolate_field(z_train, complexity_score, grid_points).reshape(cfg.grid_size, cfg.grid_size)

    boundary_curve = _extract_longest_contour(logits_grid, grid_x, grid_y)
    boundary_complexity = _interpolate_field(z_train, complexity_score, boundary_curve[:-1])

    if benchmark.boundary_points_low is None or benchmark.true_curvature_low is None:
        true_curvature = np.zeros(len(boundary_complexity), dtype=np.float32)
        alignment = float("nan")
    else:
        true_high = benchmark.embedding.transform(benchmark.boundary_points_low)
        true_projected = _project_points(true_high, projection_P, projection_mean)
        nn_true = NearestNeighbors(n_neighbors=1).fit(true_projected)
        nearest = nn_true.kneighbors(boundary_curve[:-1], return_distance=False)[:, 0]
        true_curvature = benchmark.true_curvature_low[nearest].astype(np.float32)
        if hasattr(benchmark.embedding, "W"):
            alignment = _subspace_alignment_score(np.asarray(benchmark.embedding.W, dtype=np.float32), projection_P)
        else:
            alignment = float("nan")

    threshold = float(np.quantile(boundary_complexity, cfg.high_complexity_quantile))
    chunk_labels = (boundary_complexity >= threshold).astype(np.int64)
    corr = spearmanr(boundary_complexity, true_curvature).correlation
    complexity_correlation = float(0.0 if corr is None or np.isnan(corr) else corr)

    return TeacherBoundaryProfile(
        projection_method=cfg.projection_method,
        projection_P=projection_P.astype(np.float32),
        projection_mean=projection_mean.astype(np.float32),
        projected_train=z_train.astype(np.float32),
        projected_val=z_val.astype(np.float32),
        projected_test=z_test.astype(np.float32),
        projected_logits_grid=logits_grid.astype(np.float32),
        projected_complexity_field=complexity_grid.astype(np.float32),
        grid_x=grid_x.astype(np.float32),
        grid_y=grid_y.astype(np.float32),
        teacher_boundary_curve=boundary_curve.astype(np.float32),
        teacher_boundary_complexity=boundary_complexity.astype(np.float32),
        teacher_boundary_true_curvature=true_curvature.astype(np.float32),
        boundary_chunk_labels=chunk_labels.astype(np.int64),
        complexity_correlation=complexity_correlation,
        projection_subspace_alignment=float(alignment),
    )


def _point_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-10:
        return np.linalg.norm(points - start[None, :], axis=1)
    t = ((points - start[None, :]) @ segment) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = start[None, :] + t[:, None] * segment[None, :]
    return np.linalg.norm(points - proj, axis=1)


def _polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0][None, :], axis=1)
    dists = []
    for idx in range(len(polyline) - 1):
        dists.append(_point_segment_distance(points, polyline[idx], polyline[idx + 1]))
    return np.min(np.stack(dists, axis=1), axis=1)


def _douglas_peucker(points: np.ndarray, tolerance: float) -> np.ndarray:
    if len(points) <= 2:
        return points.astype(np.float32)

    distances = _point_segment_distance(points[1:-1], points[0], points[-1])
    max_idx = int(np.argmax(distances))
    max_dist = float(distances[max_idx]) if len(distances) > 0 else 0.0
    if max_dist <= tolerance:
        return np.vstack([points[0], points[-1]]).astype(np.float32)

    split_idx = max_idx + 1
    left = _douglas_peucker(points[: split_idx + 1], tolerance)
    right = _douglas_peucker(points[split_idx:], tolerance)
    return np.vstack([left[:-1], right]).astype(np.float32)


def _fit_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(points), dtype=np.float32)])
    b = x * x + y * y
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    center = sol[:2].astype(np.float32)
    radius = float(math.sqrt(max(sol[2] + center @ center, 1e-8)))
    return center, radius


def _sample_chunk_curve(chunk: BoundaryChunk, n_points: int) -> np.ndarray:
    if chunk.sampled_curve is not None:
        if len(chunk.sampled_curve) == n_points:
            return chunk.sampled_curve.astype(np.float32)
    if chunk.kind == "polyline":
        polyline = np.asarray(chunk.polyline_points, dtype=np.float32)
        if len(polyline) <= 2:
            return np.linspace(polyline[0], polyline[-1], n_points, dtype=np.float32)
        seg_lens = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        total = float(seg_lens.sum())
        if total <= 1e-8:
            return np.repeat(polyline[:1], n_points, axis=0).astype(np.float32)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens) / total])
        t = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
        x = np.interp(t, cum, polyline[:, 0])
        y = np.interp(t, cum, polyline[:, 1])
        return np.column_stack([x, y]).astype(np.float32)
    if chunk.kind == "arc":
        theta = np.linspace(chunk.arc_start_angle, chunk.arc_end_angle, n_points, dtype=np.float32)
        x = chunk.arc_center[0] + chunk.arc_radius * np.cos(theta)
        y = chunk.arc_center[1] + chunk.arc_radius * np.sin(theta)
        return np.column_stack([x, y]).astype(np.float32)
    if chunk.kind == "spline":
        t = np.linspace(0.0, 1.0, len(chunk.spline_u_knots), dtype=np.float32)
        spline_u = CubicSpline(t, chunk.spline_u_knots.astype(np.float32))
        spline_v = CubicSpline(t, chunk.spline_v_knots.astype(np.float32))
        t_eval = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
        u = spline_u(t_eval).astype(np.float32)
        v = spline_v(t_eval).astype(np.float32)
        tangent = chunk.frame_tangent / np.linalg.norm(chunk.frame_tangent)
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
        return (
            chunk.frame_origin[None, :]
            + u[:, None] * tangent[None, :]
            + v[:, None] * normal[None, :]
        ).astype(np.float32)
    raise ValueError(f"Unsupported chunk kind {chunk.kind!r}")


def _fit_low_chunk(
    chunk_id: int,
    points: np.ndarray,
    *,
    tolerance: float,
    mean_complexity: float,
    mean_true_curvature: float,
    is_high_complexity: bool,
    arc_fit_gain: float,
    sample_points: int,
) -> BoundaryChunk:
    simplified = _douglas_peucker(points, tolerance)
    polyline_err = float(_polyline_distance(points, simplified).mean())

    if len(points) >= 4:
        center, radius = _fit_circle(points)
        circle_err = float(np.abs(np.linalg.norm(points - center[None, :], axis=1) - radius).mean())
        if circle_err < arc_fit_gain * polyline_err:
            angles = np.unwrap(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
            arc = BoundaryChunk(
                chunk_id=chunk_id,
                kind="arc",
                is_high_complexity=is_high_complexity,
                mean_complexity=float(mean_complexity),
                mean_true_curvature=float(mean_true_curvature),
                arc_center=center.astype(np.float32),
                arc_radius=float(radius),
                arc_start_angle=float(angles[0]),
                arc_end_angle=float(angles[-1]),
                trainable_params=6,
                stored_numbers=6,
            )
            sampled = _sample_chunk_curve(arc, sample_points)
            return BoundaryChunk(**{**arc.__dict__, "sampled_curve": sampled})

    poly = BoundaryChunk(
        chunk_id=chunk_id,
        kind="polyline",
        is_high_complexity=is_high_complexity,
        mean_complexity=float(mean_complexity),
        mean_true_curvature=float(mean_true_curvature),
        polyline_points=simplified.astype(np.float32),
        trainable_params=int(2 * simplified.size // 2),
        stored_numbers=int(2 * simplified.size // 2),
    )
    sampled = _sample_chunk_curve(poly, sample_points)
    return BoundaryChunk(**{**poly.__dict__, "sampled_curve": sampled})


def _fit_spline_chunk(
    chunk_id: int,
    points: np.ndarray,
    *,
    n_knots: int,
    mean_complexity: float,
    mean_true_curvature: float,
    sample_points: int,
) -> BoundaryChunk:
    origin = points[0].astype(np.float32)
    direction = points[-1] - points[0]
    if np.linalg.norm(direction) <= 1e-6:
        centered = points - points.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
    tangent = direction / np.linalg.norm(direction)
    tangent = tangent.astype(np.float32)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    local = np.column_stack(
        [
            (points - origin[None, :]) @ tangent,
            (points - origin[None, :]) @ normal,
        ]
    ).astype(np.float32)
    seg = np.linalg.norm(np.diff(local, axis=0), axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seg)])
    if arclen[-1] <= 1e-8:
        arclen = np.linspace(0.0, 1.0, len(local), dtype=np.float32)
    else:
        arclen = (arclen / arclen[-1]).astype(np.float32)
    t_knots = np.linspace(0.0, 1.0, n_knots, dtype=np.float32)
    u_knots = np.interp(t_knots, arclen, local[:, 0]).astype(np.float32)
    v_knots = np.interp(t_knots, arclen, local[:, 1]).astype(np.float32)
    spline = BoundaryChunk(
        chunk_id=chunk_id,
        kind="spline",
        is_high_complexity=True,
        mean_complexity=float(mean_complexity),
        mean_true_curvature=float(mean_true_curvature),
        frame_origin=origin,
        frame_tangent=tangent,
        spline_u_knots=u_knots,
        spline_v_knots=v_knots,
        trainable_params=int(4 + 2 * n_knots),
        stored_numbers=int(4 + 2 * n_knots),
    )
    sampled = _sample_chunk_curve(spline, sample_points)
    return BoundaryChunk(**{**spline.__dict__, "sampled_curve": sampled})


def _merge_small_chunks(chunks: list[dict[str, Any]], min_points: int) -> list[dict[str, Any]]:
    if len(chunks) <= 1:
        return chunks
    merged = chunks[:]
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for idx, chunk in enumerate(list(merged)):
            if len(chunk["points"]) >= min_points:
                continue
            left = (idx - 1) % len(merged)
            right = (idx + 1) % len(merged)
            if len(merged[left]["points"]) <= len(merged[right]["points"]):
                target = left
            else:
                target = right
            target_points = merged[target]["points"]
            target_n = len(target_points)
            chunk_n = len(chunk["points"])
            if target == left:
                new_points = np.vstack([target_points, chunk["points"]]).astype(np.float32)
            else:
                new_points = np.vstack([chunk["points"], target_points]).astype(np.float32)
            merged[target]["points"] = new_points
            merged[target]["mean_complexity"] = float(
                (target_n * merged[target]["mean_complexity"] + chunk_n * chunk["mean_complexity"])
                / max(target_n + chunk_n, 1)
            )
            merged[target]["mean_true_curvature"] = float(
                (target_n * merged[target]["mean_true_curvature"] + chunk_n * chunk["mean_true_curvature"])
                / max(target_n + chunk_n, 1)
            )
            merged[target]["is_high_complexity"] = bool(
                merged[target]["is_high_complexity"] or chunk["is_high_complexity"]
            )
            del merged[idx]
            changed = True
            break
    return merged


def _split_boundary_chunks(
    boundary_curve: np.ndarray,
    high_mask: np.ndarray,
    boundary_complexity: np.ndarray,
    boundary_true_curvature: np.ndarray,
    *,
    min_points: int,
) -> list[dict[str, Any]]:
    curve = boundary_curve[:-1].astype(np.float32)
    mask = high_mask.astype(bool)
    if len(curve) == 0:
        return []
    if np.all(mask == mask[0]):
        return [
            {
                "points": curve,
                "is_high_complexity": bool(mask[0]),
                "mean_complexity": float(boundary_complexity.mean()),
                "mean_true_curvature": float(boundary_true_curvature.mean()),
            }
        ]
    changes = np.flatnonzero(mask != np.roll(mask, 1))
    start = int(changes[0])
    curve = np.concatenate([curve[start:], curve[:start]], axis=0)
    mask = np.concatenate([mask[start:], mask[:start]], axis=0)
    complexity = np.concatenate([boundary_complexity[start:], boundary_complexity[:start]], axis=0)
    true_curv = np.concatenate([boundary_true_curvature[start:], boundary_true_curvature[:start]], axis=0)
    boundaries = np.flatnonzero(mask != np.roll(mask, -1)) + 1
    split_points = [0] + boundaries.tolist() + [len(curve)]
    chunks = []
    for idx in range(len(split_points) - 1):
        left, right = split_points[idx], split_points[idx + 1]
        pts = curve[left:right]
        if len(pts) == 0:
            continue
        chunks.append(
            {
                "points": pts.astype(np.float32),
                "is_high_complexity": bool(mask[left]),
                "mean_complexity": float(complexity[left:right].mean()),
                "mean_true_curvature": float(true_curv[left:right].mean()),
            }
        )
    return _merge_small_chunks(chunks, min_points=min_points)


def _build_boundary_polygon(chunks: list[BoundaryChunk]) -> np.ndarray:
    sampled = []
    for idx, chunk in enumerate(chunks):
        pts = chunk.sampled_curve if chunk.sampled_curve is not None else _sample_chunk_curve(chunk, 64)
        if idx > 0:
            pts = pts[1:]
        sampled.append(pts)
    polygon = np.vstack(sampled).astype(np.float32)
    if np.linalg.norm(polygon[0] - polygon[-1]) > 1e-6:
        polygon = np.vstack([polygon, polygon[:1]])
    return polygon.astype(np.float32)


def _signed_distance_to_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    path = Path(polygon)
    inside = path.contains_points(points)
    seg_starts = polygon[:-1]
    seg_ends = polygon[1:]
    min_dists = np.full(len(points), np.inf, dtype=np.float32)
    for start, end in zip(seg_starts, seg_ends):
        d = _point_segment_distance(points, start, end).astype(np.float32)
        min_dists = np.minimum(min_dists, d)
    sign = np.where(inside, 1.0, -1.0).astype(np.float32)
    return sign * min_dists


def _calibrate_boundary_logits(
    polygon: np.ndarray,
    projected_val: np.ndarray,
    y_val: np.ndarray,
    *,
    steps: int,
    lr: float,
) -> tuple[float, float]:
    distances = torch.tensor(_signed_distance_to_polygon(projected_val, polygon), dtype=torch.float32).unsqueeze(1)
    targets = torch.tensor(y_val.astype(np.float32), dtype=torch.float32).unsqueeze(1)
    log_temp = torch.nn.Parameter(torch.tensor([math.log(8.0)], dtype=torch.float32))
    bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    optimizer = torch.optim.Adam([log_temp, bias], lr=lr)
    best = (math.inf, 8.0, 0.0)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = distances * torch.exp(log_temp) + bias
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        optimizer.step()
        val = float(loss.detach().item())
        if val < best[0]:
            best = (val, float(torch.exp(log_temp).item()), float(bias.item()))
    return best[1], best[2]


def _sigmoid_probs(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _evaluate_boundary_model(
    model: HybridBoundaryModel,
    benchmark: EmbeddedClassificationBenchmark,
    *,
    split: str,
) -> dict[str, float]:
    data = getattr(benchmark, split)
    projected = _project_points(data.X_high, model.projection_P, model.projection_mean)
    signed_distance = _signed_distance_to_polygon(projected, model.boundary_polygon)
    logits = model.temperature * signed_distance + model.bias
    probs = _sigmoid_probs(logits)
    y = data.y.astype(np.float32)
    eps = 1e-6
    bce = -np.mean(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
    preds = (probs > 0.5).astype(np.float32)
    return {
        "bce": float(bce),
        "accuracy": float((preds == y).mean()),
        "mean_margin": float(np.mean(np.abs(logits))),
    }


def _count_model_storage(
    projection_P: np.ndarray,
    projection_mean: np.ndarray,
    chunks: list[BoundaryChunk],
) -> tuple[int, int]:
    trainable = 2 + int(sum(chunk.trainable_params for chunk in chunks))
    stored = trainable + int(projection_P.size + projection_mean.size)
    return trainable, stored


def build_hybrid_boundary_model(
    profile: TeacherBoundaryProfile,
    benchmark: EmbeddedClassificationBenchmark,
    cfg: BoundaryCompressionConfig,
    *,
    target_budget: int,
) -> HybridBoundaryModel:
    chunks_meta = _split_boundary_chunks(
        profile.teacher_boundary_curve,
        profile.boundary_chunk_labels.astype(bool),
        profile.teacher_boundary_complexity,
        profile.teacher_boundary_true_curvature,
        min_points=cfg.chunk_min_points,
    )
    tolerance = cfg.low_chunk_tolerance
    final_chunks: list[BoundaryChunk] = []
    selected_name = "tight"

    while True:
        fallback_chunks: list[BoundaryChunk] = []
        for chunk_id, chunk in enumerate(chunks_meta):
            tol = tolerance * (0.85 if chunk["is_high_complexity"] else 1.0)
            fallback_chunks.append(
                _fit_low_chunk(
                    chunk_id,
                    chunk["points"],
                    tolerance=tol,
                    mean_complexity=chunk["mean_complexity"],
                    mean_true_curvature=chunk["mean_true_curvature"],
                    is_high_complexity=chunk["is_high_complexity"],
                    arc_fit_gain=cfg.arc_fit_gain,
                    sample_points=cfg.chunk_sample_points,
                )
            )
        fallback_trainable = 2 + sum(chunk.trainable_params for chunk in fallback_chunks)
        if fallback_trainable <= target_budget or tolerance >= cfg.low_chunk_max_tolerance:
            final_chunks = fallback_chunks
            break
        tolerance *= cfg.low_chunk_tolerance_growth

    remaining = max(target_budget - (2 + sum(chunk.trainable_params for chunk in final_chunks)), 0)
    high_ids = [
        idx for idx, (meta, chunk) in enumerate(zip(chunks_meta, final_chunks))
        if meta["is_high_complexity"] and len(meta["points"]) >= cfg.min_spline_knots
    ]
    high_ids.sort(key=lambda idx: chunks_meta[idx]["mean_complexity"], reverse=True)
    selected_splines: dict[int, int] = {}
    for idx in high_ids:
        min_cost = 4 + 2 * cfg.min_spline_knots
        extra_cost = max(min_cost - final_chunks[idx].trainable_params, 0)
        if remaining >= extra_cost:
            selected_splines[idx] = cfg.min_spline_knots
            remaining -= extra_cost
    if selected_splines and remaining >= 2:
        weights = np.asarray([chunks_meta[idx]["mean_complexity"] for idx in selected_splines], dtype=np.float32)
        weights = weights / max(weights.sum(), 1e-8)
        extra_knots = remaining // 2
        for idx, weight in zip(selected_splines, weights):
            add = int(round(extra_knots * float(weight)))
            selected_splines[idx] = min(cfg.max_spline_knots, selected_splines[idx] + add)
    upgraded_chunks = final_chunks[:]
    for idx, knot_count in selected_splines.items():
        upgraded_chunks[idx] = _fit_spline_chunk(
            idx,
            chunks_meta[idx]["points"],
            n_knots=max(cfg.min_spline_knots, knot_count),
            mean_complexity=chunks_meta[idx]["mean_complexity"],
            mean_true_curvature=chunks_meta[idx]["mean_true_curvature"],
            sample_points=cfg.chunk_sample_points,
        )

    boundary_polygon = _build_boundary_polygon(upgraded_chunks)
    temperature, bias = _calibrate_boundary_logits(
        boundary_polygon,
        profile.projected_val,
        benchmark.val.y,
        steps=cfg.calibration_steps,
        lr=cfg.calibration_lr,
    )
    trainable, stored = _count_model_storage(profile.projection_P, profile.projection_mean, upgraded_chunks)
    low_chunks = tuple(chunk for chunk in upgraded_chunks if chunk.kind != "spline")
    high_chunks = tuple(chunk for chunk in upgraded_chunks if chunk.kind == "spline")
    return HybridBoundaryModel(
        projection_P=profile.projection_P.astype(np.float32),
        projection_mean=profile.projection_mean.astype(np.float32),
        low_curvature_chunks=low_chunks,
        high_curvature_chunks=high_chunks,
        boundary_polygon=boundary_polygon.astype(np.float32),
        temperature=float(temperature),
        bias=float(bias),
        trainable_params=int(trainable),
        stored_numbers=int(stored),
        target_budget=int(target_budget),
        selected_budget_name=selected_name,
    )


def _fit_global_polyline_model(
    profile: TeacherBoundaryProfile,
    benchmark: EmbeddedClassificationBenchmark,
    cfg: BoundaryCompressionConfig,
    *,
    target_budget: int,
) -> HybridBoundaryModel:
    curve = profile.teacher_boundary_curve[:-1]
    tolerance = cfg.low_chunk_tolerance
    while True:
        simplified = _douglas_peucker(curve, tolerance)
        trainable = 2 + int(2 * len(simplified))
        if trainable <= target_budget or tolerance >= cfg.low_chunk_max_tolerance:
            break
        tolerance *= cfg.low_chunk_tolerance_growth
    chunk = BoundaryChunk(
        chunk_id=0,
        kind="polyline",
        is_high_complexity=False,
        mean_complexity=float(profile.teacher_boundary_complexity.mean()),
        mean_true_curvature=float(profile.teacher_boundary_true_curvature.mean()),
        polyline_points=simplified.astype(np.float32),
        sampled_curve=_sample_chunk_curve(
            BoundaryChunk(
                chunk_id=0,
                kind="polyline",
                is_high_complexity=False,
                mean_complexity=float(profile.teacher_boundary_complexity.mean()),
                mean_true_curvature=float(profile.teacher_boundary_true_curvature.mean()),
                polyline_points=simplified.astype(np.float32),
                trainable_params=int(2 * len(simplified)),
                stored_numbers=int(2 * len(simplified)),
            ),
            cfg.chunk_sample_points * 2,
        ),
        trainable_params=int(2 * len(simplified)),
        stored_numbers=int(2 * len(simplified)),
    )
    polygon = _build_boundary_polygon([chunk])
    temperature, bias = _calibrate_boundary_logits(
        polygon,
        profile.projected_val,
        benchmark.val.y,
        steps=cfg.calibration_steps,
        lr=cfg.calibration_lr,
    )
    trainable, stored = _count_model_storage(profile.projection_P, profile.projection_mean, [chunk])
    return HybridBoundaryModel(
        projection_P=profile.projection_P.astype(np.float32),
        projection_mean=profile.projection_mean.astype(np.float32),
        low_curvature_chunks=(chunk,),
        high_curvature_chunks=(),
        boundary_polygon=polygon.astype(np.float32),
        temperature=float(temperature),
        bias=float(bias),
        trainable_params=int(trainable),
        stored_numbers=int(stored),
        target_budget=int(target_budget),
        selected_budget_name="polyline_only",
    )


def _fit_global_spline_model(
    profile: TeacherBoundaryProfile,
    benchmark: EmbeddedClassificationBenchmark,
    cfg: BoundaryCompressionConfig,
    *,
    target_budget: int,
) -> HybridBoundaryModel:
    available = max(target_budget - 2 - 4, 2 * cfg.min_spline_knots)
    n_knots = max(cfg.min_spline_knots, min(cfg.max_spline_knots, available // 2))
    curve = profile.teacher_boundary_curve[:-1]
    arclen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
    if arclen[-1] <= 1e-8:
        arclen = np.linspace(0.0, 1.0, len(curve), dtype=np.float32)
    else:
        arclen = (arclen / arclen[-1]).astype(np.float32)
    t_knots = np.linspace(0.0, 1.0, n_knots, dtype=np.float32)
    x_knots = np.interp(t_knots, arclen, curve[:, 0]).astype(np.float32)
    y_knots = np.interp(t_knots, arclen, curve[:, 1]).astype(np.float32)
    origin = np.zeros(2, dtype=np.float32)
    tangent = np.asarray([1.0, 0.0], dtype=np.float32)
    chunk = BoundaryChunk(
        chunk_id=0,
        kind="spline",
        is_high_complexity=True,
        mean_complexity=float(profile.teacher_boundary_complexity.mean()),
        mean_true_curvature=float(profile.teacher_boundary_true_curvature.mean()),
        frame_origin=origin,
        frame_tangent=tangent,
        spline_u_knots=x_knots,
        spline_v_knots=y_knots,
        trainable_params=int(4 + 2 * n_knots),
        stored_numbers=int(4 + 2 * n_knots),
    )
    sampled = _sample_chunk_curve(chunk, cfg.chunk_sample_points * 3)
    chunk = BoundaryChunk(**{**chunk.__dict__, "sampled_curve": sampled})
    polygon = sampled
    if np.linalg.norm(polygon[0] - polygon[-1]) > 1e-6:
        polygon = np.vstack([polygon, polygon[:1]])
    temperature, bias = _calibrate_boundary_logits(
        polygon,
        profile.projected_val,
        benchmark.val.y,
        steps=cfg.calibration_steps,
        lr=cfg.calibration_lr,
    )
    trainable, stored = _count_model_storage(profile.projection_P, profile.projection_mean, [chunk])
    return HybridBoundaryModel(
        projection_P=profile.projection_P.astype(np.float32),
        projection_mean=profile.projection_mean.astype(np.float32),
        low_curvature_chunks=(),
        high_curvature_chunks=(chunk,),
        boundary_polygon=polygon.astype(np.float32),
        temperature=float(temperature),
        bias=float(bias),
        trainable_params=int(trainable),
        stored_numbers=int(stored),
        target_budget=int(target_budget),
        selected_budget_name="spline_only",
    )


def _fit_boundary_budget_pair(
    profile: TeacherBoundaryProfile,
    benchmark: EmbeddedClassificationBenchmark,
    cfg: BoundaryCompressionConfig,
    *,
    small_reference_params: int,
) -> tuple[HybridBoundaryModel, HybridBoundaryModel, HybridBoundaryModel]:
    tight_budget = max(
        cfg.min_trainable_budget,
        min(cfg.target_trainable_cap, small_reference_params // max(cfg.target_compression_ratio, 1)),
    )
    fallback_budget = max(
        tight_budget,
        min(cfg.fallback_trainable_cap, small_reference_params // max(cfg.fallback_compression_ratio, 1)),
    )
    hybrid_tight = build_hybrid_boundary_model(profile, benchmark, cfg, target_budget=tight_budget)
    hybrid_fallback = build_hybrid_boundary_model(profile, benchmark, cfg, target_budget=fallback_budget)
    tight_val = _evaluate_boundary_model(hybrid_tight, benchmark, split="val")
    fallback_val = _evaluate_boundary_model(hybrid_fallback, benchmark, split="val")
    chosen = hybrid_tight if tight_val["bce"] <= fallback_val["bce"] else hybrid_fallback
    chosen_name = "tight" if chosen is hybrid_tight else "fallback"
    chosen = HybridBoundaryModel(**{**chosen.__dict__, "selected_budget_name": chosen_name})
    polyline = _fit_global_polyline_model(profile, benchmark, cfg, target_budget=chosen.target_budget)
    spline = _fit_global_spline_model(profile, benchmark, cfg, target_budget=chosen.target_budget)
    return chosen, polyline, spline


def _fit_tiny_relu_under_budget(
    input_dim: int,
    target_budget: int,
) -> nn.Module:
    candidates: list[tuple[int, int]] = []
    for depth in (1, 2, 3):
        for hidden in range(1, 65):
            params = _mlp_param_count(input_dim, hidden, depth)
            if params <= target_budget:
                candidates.append((params, hidden, depth))
    if candidates:
        _, hidden, depth = max(candidates, key=lambda item: (item[0], -item[1]))
        return _make_classifier(input_dim, hidden, depth)
    if input_dim + 1 <= target_budget:
        return nn.Linear(input_dim, 1)
    return _ConstantLogit()


def _bucket_metrics(values: np.ndarray, scores: np.ndarray, y: np.ndarray) -> dict[str, dict[str, float]]:
    q1, q2 = np.quantile(scores, [0.33, 0.66]).tolist()
    buckets = {
        "low": scores <= q1,
        "mid": (scores > q1) & (scores <= q2),
        "high": scores > q2,
    }
    out = {}
    probs = _sigmoid_probs(values)
    preds = (probs > 0.5).astype(np.float32)
    for name, mask in buckets.items():
        if not np.any(mask):
            out[name] = {"accuracy": float("nan"), "bce": float("nan"), "count": 0.0}
            continue
        yy = y[mask].astype(np.float32)
        pp = probs[mask]
        eps = 1e-6
        bce = -np.mean(yy * np.log(pp + eps) + (1.0 - yy) * np.log(1.0 - pp + eps))
        out[name] = {
            "accuracy": float((preds[mask] == yy).mean()),
            "bce": float(bce),
            "count": float(mask.sum()),
        }
    return out


def _test_bucket_scores(
    benchmark: EmbeddedClassificationBenchmark,
    profile: TeacherBoundaryProfile,
    model: HybridBoundaryModel,
) -> dict[str, Any]:
    projected_test = _project_points(benchmark.test.X_high, model.projection_P, model.projection_mean)
    logits = model.temperature * _signed_distance_to_polygon(projected_test, model.boundary_polygon) + model.bias
    if benchmark.boundary_points_low is None or benchmark.true_curvature_low is None:
        return {}

    nn_true = NearestNeighbors(n_neighbors=1).fit(benchmark.boundary_points_low)
    nearest_true = nn_true.kneighbors(benchmark.test.X_low, return_distance=False)[:, 0]
    true_scores = benchmark.true_curvature_low[nearest_true]

    teacher_proj = profile.teacher_boundary_curve[:-1]
    nn_prof = NearestNeighbors(n_neighbors=1).fit(teacher_proj)
    nearest_prof = nn_prof.kneighbors(projected_test, return_distance=False)[:, 0]
    teacher_scores = profile.teacher_boundary_complexity[nearest_prof]

    return {
        "true_curvature": _bucket_metrics(logits, true_scores, benchmark.test.y),
        "teacher_complexity": _bucket_metrics(logits, teacher_scores, benchmark.test.y),
    }


def _boundary_model_metrics(
    model: HybridBoundaryModel,
    benchmark: EmbeddedClassificationBenchmark,
    *,
    profile: TeacherBoundaryProfile,
) -> dict[str, Any]:
    val = _evaluate_boundary_model(model, benchmark, split="val")
    test = _evaluate_boundary_model(model, benchmark, split="test")
    return {
        "trainable_params": float(model.trainable_params),
        "stored_numbers": float(model.stored_numbers),
        "compression_ratio_vs_small_32x4": float("nan"),
        "val_bce": float(val["bce"]),
        "val_accuracy": float(val["accuracy"]),
        "test_bce": float(test["bce"]),
        "test_accuracy": float(test["accuracy"]),
        "test_margin": float(test["mean_margin"]),
        "n_low_chunks": float(len(model.low_curvature_chunks)),
        "n_high_chunks": float(len(model.high_curvature_chunks)),
        "bucket_metrics": _test_bucket_scores(benchmark, profile, model),
    }


def run_single_boundary_compression(
    cfg: BoundaryCompressionConfig,
    *,
    dataset: str,
    observed_dim: int,
) -> dict[str, Any]:
    if dataset == "multi_island":
        raise ValueError(
            "The current boundary compressor assumes a single connected contour; "
            "'multi_island' is available at the dataset layer only."
        )
    _set_seed(cfg.seed)
    device = _device_for(cfg)
    benchmark = make_embedded_boundary_classification_benchmark(
        dataset=dataset,
        observed_dim=observed_dim,
        n_samples=cfg.n_samples,
        noise=cfg.noise,
        boundary_fraction=cfg.boundary_fraction,
        n_boundary_points=cfg.n_boundary_points,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        random_state=cfg.seed,
    )

    X_train = _tensor(benchmark.train.X_high, device)
    y_train = _targets(benchmark.train.y, device)
    X_val = _tensor(benchmark.val.X_high, device)
    y_val = _targets(benchmark.val.y, device)
    X_test = _tensor(benchmark.test.X_high, device)
    y_test = _targets(benchmark.test.y, device)

    source_model = _make_classifier(benchmark.observed_dim, cfg.source_hidden, cfg.source_depth).to(device)
    source_start = time.time()
    source_trace, source_best = _train_classifier_with_best_checkpoint(
        source_model,
        X_train,
        y_train,
        epochs=cfg.source_epochs,
        lr=cfg.source_lr,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        trace_every_n_steps=cfg.trace_every_n_steps,
    )
    source_time = time.time() - source_start
    source_metrics = _collect_model_metrics(source_model, benchmark, device=device)
    source_metrics["train_time_s"] = source_time

    small_model = _make_classifier(benchmark.observed_dim, cfg.small_hidden, cfg.small_depth).to(device)
    small_trace, small_best = _train_classifier_with_best_checkpoint(
        small_model,
        X_train,
        y_train,
        epochs=cfg.baseline_epochs,
        lr=cfg.baseline_lr,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        trace_every_n_steps=cfg.trace_every_n_steps,
    )
    small_metrics = _collect_model_metrics(small_model, benchmark, device=device)

    gate_passed = bool(
        (source_metrics["val_accuracy"] - small_metrics["val_accuracy"]) >= cfg.gate_accuracy_margin
        or source_metrics["val_bce"] <= cfg.gate_bce_ratio * small_metrics["val_bce"]
    )

    profile = build_teacher_boundary_profile(source_model, benchmark, cfg, device=device)
    hybrid_model, polyline_model, spline_model = _fit_boundary_budget_pair(
        profile,
        benchmark,
        cfg,
        small_reference_params=int(small_metrics["params"]),
    )

    ambient_tiny = _fit_tiny_relu_under_budget(benchmark.observed_dim, hybrid_model.trainable_params).to(device)
    ambient_trace, ambient_best = _train_classifier_with_best_checkpoint(
        ambient_tiny,
        X_train,
        y_train,
        epochs=cfg.baseline_epochs,
        lr=cfg.baseline_lr,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        trace_every_n_steps=cfg.trace_every_n_steps,
    )
    ambient_metrics = _collect_model_metrics(ambient_tiny, benchmark, device=device)

    projected_train = _tensor(profile.projected_train, device)
    projected_val = _tensor(profile.projected_val, device)
    projected_test = _tensor(profile.projected_test, device)
    projected_tiny = _fit_tiny_relu_under_budget(cfg.projection_rank, hybrid_model.trainable_params).to(device)
    projected_trace, projected_best = _train_classifier_with_best_checkpoint(
        projected_tiny,
        projected_train,
        y_train,
        epochs=cfg.baseline_epochs,
        lr=cfg.baseline_lr,
        X_val=projected_val,
        y_val=y_val,
        X_test=projected_test,
        y_test=y_test,
        trace_every_n_steps=cfg.trace_every_n_steps,
    )
    projected_val_metrics = _eval_classifier(projected_tiny, projected_val, y_val)
    projected_test_metrics = _eval_classifier(projected_tiny, projected_test, y_test)
    projected_metrics = {
        "params": float(_num_params(projected_tiny)),
        "val_bce": float(projected_val_metrics["bce"]),
        "val_accuracy": float(projected_val_metrics["accuracy"]),
        "test_bce": float(projected_test_metrics["bce"]),
        "test_accuracy": float(projected_test_metrics["accuracy"]),
        "test_margin": float(projected_test_metrics["mean_margin"]),
    }

    hybrid_metrics = _boundary_model_metrics(hybrid_model, benchmark, profile=profile)
    polyline_metrics = _boundary_model_metrics(polyline_model, benchmark, profile=profile)
    spline_metrics = _boundary_model_metrics(spline_model, benchmark, profile=profile)
    for item in (hybrid_metrics, polyline_metrics, spline_metrics):
        item["compression_ratio_vs_small_32x4"] = float(small_metrics["params"] / item["trainable_params"])

    return {
        "config": _serialize(cfg),
        "benchmark": {
            "name": benchmark.name,
            "observed_dim": benchmark.observed_dim,
            "intrinsic_dim": benchmark.intrinsic_dim,
            "train_size": len(benchmark.train.y),
            "val_size": len(benchmark.val.y),
            "test_size": len(benchmark.test.y),
        },
        "gate": {
            "passed": gate_passed,
            "source_val_accuracy": float(source_metrics["val_accuracy"]),
            "small_val_accuracy": float(small_metrics["val_accuracy"]),
            "source_val_bce": float(source_metrics["val_bce"]),
            "small_val_bce": float(small_metrics["val_bce"]),
            "source_minus_small_accuracy": float(source_metrics["val_accuracy"] - small_metrics["val_accuracy"]),
            "source_over_small_bce_ratio": float(
                source_metrics["val_bce"] / max(small_metrics["val_bce"], 1e-8)
            ),
        },
        "teacher_profile": _serialize(profile),
        "metrics": {
            "source_mlp": source_metrics,
            "small_relu_32x4": small_metrics,
            "ambient_tiny_relu": ambient_metrics,
            "projected_tiny_relu": projected_metrics,
            "projected_global_polyline": polyline_metrics,
            "projected_global_spline": spline_metrics,
            "hybrid_boundary": hybrid_metrics,
        },
        "models": {
            "hybrid_boundary": _serialize(hybrid_model),
            "projected_global_polyline": _serialize(polyline_model),
            "projected_global_spline": _serialize(spline_model),
        },
        "training_traces": {
            "source_mlp": source_trace,
            "small_relu_32x4": small_trace,
            "ambient_tiny_relu": ambient_trace,
            "projected_tiny_relu": projected_trace,
        },
        "best_checkpoints": {
            "source_mlp": source_best,
            "small_relu_32x4": small_best,
            "ambient_tiny_relu": ambient_best,
            "projected_tiny_relu": projected_best,
        },
    }


def gate_boundary_task(
    cfg: BoundaryCompressionConfig,
    *,
    dataset: str,
) -> dict[str, Any]:
    result = run_single_boundary_compression(cfg, dataset=dataset, observed_dim=cfg.gate_observed_dim)
    return {
        "dataset": dataset,
        "observed_dim": cfg.gate_observed_dim,
        "passed": bool(result["gate"]["passed"]),
        "gate": result["gate"],
    }


def run_boundary_compression_program(cfg: BoundaryCompressionConfig) -> dict[str, Any]:
    gate_results = {}
    retained = []
    for dataset in cfg.datasets:
        gate_result = gate_boundary_task(cfg, dataset=dataset)
        gate_results[dataset] = gate_result
        if gate_result["passed"]:
            retained.append(dataset)

    results: dict[str, Any] = {}
    for dataset in retained:
        results[dataset] = {}
        for observed_dim in cfg.observed_dims:
            results[dataset][str(observed_dim)] = run_single_boundary_compression(
                cfg,
                dataset=dataset,
                observed_dim=observed_dim,
            )
    return {
        "program": "curvature_adaptive_boundary_compression",
        "retained_datasets": retained,
        "gate_results": gate_results,
        "results": results,
    }


def save_boundary_compression_results(results: dict[str, Any], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
