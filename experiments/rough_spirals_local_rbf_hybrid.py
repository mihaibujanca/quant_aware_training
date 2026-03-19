#!/usr/bin/env python3
"""Projected RBF hybrid compression on rough spirals."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from aleph.datasets import make_embedded_classification_benchmark
from aleph.differential import jacobian_variation
from aleph.relu_geometric_compression import (
    _device_for,
    _eval_classifier,
    _make_classifier,
    _set_seed,
    _targets,
    _tensor,
    HighDReluCompressionConfig,
)
from experiments.rough_spirals_local_patch_hybrid import (
    _ProjectedTeacherView,
    _fit_classifier_best,
    _normalize,
    _projection_from_teacher,
    _subspace_alignment_score,
    _teacher_complexity,
)


def _fit_mlp_near_budget(
    input_dim: int,
    target_budget: int,
    *,
    depth_options: tuple[int, ...] = (1, 2, 3, 4),
) -> tuple[nn.Module, dict[str, float]]:
    max_hidden = min(512, max(64, target_budget))
    best: tuple[int, int, int, int] | None = None
    for depth in depth_options:
        for hidden in range(1, max_hidden + 1):
            model = _make_classifier(input_dim, hidden, depth)
            params = sum(p.numel() for p in model.parameters())
            item = (abs(params - target_budget), -params, hidden, depth)
            if best is None or item < best:
                best = item
    if best is None:
        model = nn.Linear(input_dim, 1)
        params = sum(p.numel() for p in model.parameters())
        return model, {
            "target_budget": float(target_budget),
            "params": float(params),
            "param_gap": float(abs(params - target_budget)),
            "within_10pct": float(abs(params - target_budget) <= 0.10 * max(target_budget, 1)),
        }
    _, neg_params, hidden, depth = best
    params = -neg_params
    model = _make_classifier(input_dim, hidden, depth)
    return model, {
        "target_budget": float(target_budget),
        "params": float(params),
        "param_gap": float(abs(params - target_budget)),
        "within_10pct": float(abs(params - target_budget) <= 0.10 * max(target_budget, 1)),
        "hidden": float(hidden),
        "depth": float(depth),
    }


def _bucket_metrics(logits: np.ndarray, scores: np.ndarray, y: np.ndarray) -> dict[str, dict[str, float]]:
    q1, q2 = np.quantile(scores, [0.33, 0.66]).tolist()
    buckets = {
        "low": scores <= q1,
        "mid": (scores > q1) & (scores <= q2),
        "high": scores > q2,
    }
    out = {}
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
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


@lru_cache(maxsize=None)
def _monomial_terms(input_dim: int, degree: int) -> tuple[tuple[int, ...], ...]:
    terms: list[tuple[int, ...]] = [()]
    for current_degree in range(1, degree + 1):
        terms.extend(combinations_with_replacement(range(input_dim), current_degree))
    return tuple(terms)


def _polynomial_features(z: torch.Tensor, degree: int) -> torch.Tensor:
    terms = _monomial_terms(int(z.shape[1]), degree)
    features = []
    for term in terms:
        if not term:
            features.append(torch.ones(len(z), device=z.device, dtype=z.dtype))
            continue
        feat = torch.ones(len(z), device=z.device, dtype=z.dtype)
        for idx in term:
            feat = feat * z[:, idx]
        features.append(feat)
    return torch.stack(features, dim=1)


def _polynomial_features_np(z: np.ndarray, degree: int) -> np.ndarray:
    terms = _monomial_terms(int(z.shape[1]), degree)
    features = []
    for term in terms:
        if not term:
            features.append(np.ones(len(z), dtype=np.float32))
            continue
        feat = np.ones(len(z), dtype=np.float32)
        for idx in term:
            feat = feat * z[:, idx]
        features.append(feat.astype(np.float32))
    return np.stack(features, axis=1).astype(np.float32)


def _fit_polynomial_reference(z: np.ndarray, logits: np.ndarray, degree: int) -> np.ndarray:
    design = _polynomial_features_np(z, degree)
    coeffs, _, _, _ = np.linalg.lstsq(design, logits.reshape(-1, 1), rcond=None)
    return coeffs.astype(np.float32)


def _projected_field_complexity(
    teacher: nn.Module,
    X_high: np.ndarray,
    *,
    projection_P: np.ndarray,
    projection_mean: np.ndarray,
    poly_coeffs: np.ndarray,
    poly_degree: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    projected = ((X_high - projection_mean) @ projection_P).astype(np.float32)
    x = _tensor(X_high, device)
    logits = teacher(x).detach().squeeze(1).cpu().numpy().astype(np.float32)
    margin_inv = 1.0 / (np.abs(logits) + 1e-3)

    projected_teacher = _ProjectedTeacherView(teacher, projection_P, projection_mean).to(device)
    jac_var = jacobian_variation(
        projected_teacher,
        torch.tensor(projected, dtype=torch.float32, device=device),
        eps=0.02,
        chunk_size=256,
    ).detach().cpu().numpy().astype(np.float32)

    poly_logits = (_polynomial_features_np(projected, poly_degree) @ poly_coeffs).reshape(-1).astype(np.float32)
    residual = np.abs(logits - poly_logits).astype(np.float32)

    complexity = (
        0.45 * _normalize(jac_var)
        + 0.30 * _normalize(margin_inv)
        + 0.25 * _normalize(residual)
    ).astype(np.float32)
    return projected, complexity


def _build_atom_specs(
    projected: np.ndarray,
    complexity: np.ndarray,
    *,
    n_atoms: int,
    quantile: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    threshold = float(np.quantile(complexity, quantile))
    hard_mask = complexity >= threshold
    projected_dim = int(projected.shape[1]) if projected.ndim == 2 else 0
    if int(hard_mask.sum()) < n_atoms:
        keep = np.argsort(complexity)[-max(n_atoms * 3, 64):]
        hard_mask = np.zeros(len(projected), dtype=bool)
        hard_mask[keep] = True
    hard_points = projected[hard_mask]
    hard_weights = complexity[hard_mask]
    if len(hard_points) == 0:
        return np.zeros((0, projected_dim), dtype=np.float32), np.zeros(0, dtype=np.float32)

    n_clusters = min(n_atoms, len(hard_points))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(
        hard_points,
        sample_weight=hard_weights,
    )

    centers = []
    scales = []
    nbrs = None
    if len(hard_points) > 2:
        nbrs = NearestNeighbors(n_neighbors=min(8, len(hard_points))).fit(hard_points)

    for cluster in range(n_clusters):
        mask = labels == cluster
        pts = hard_points[mask]
        weights = hard_weights[mask]
        if len(pts) == 0:
            continue
        center = np.average(pts, axis=0, weights=weights).astype(np.float32)
        if len(pts) > 1:
            sqdist = np.sum((pts - center) ** 2, axis=1)
            scale = float(np.sqrt(np.average(sqdist, weights=weights) + 1e-5))
        elif nbrs is not None:
            dists, _ = nbrs.kneighbors(center[None, :], return_distance=True)
            scale = float(np.mean(dists[0, 1:]) + 1e-5)
        else:
            scale = 0.08
        scales.append(float(np.clip(scale * 0.9, 0.03, 0.35)))
        centers.append(center)

    if not centers:
        return np.zeros((0, projected_dim), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.stack(centers, axis=0).astype(np.float32), np.asarray(scales, dtype=np.float32)


class ProjectedRBFHybrid(nn.Module):
    def __init__(self, input_dim: int, poly_degree: int, centers: np.ndarray, scales: np.ndarray):
        super().__init__()
        self.input_dim = int(input_dim)
        self.poly_degree = int(poly_degree)
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.float32))
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))

        backbone_dim = len(_monomial_terms(self.input_dim, self.poly_degree))
        self.global_weights = nn.Parameter(torch.zeros(backbone_dim))
        self.local_weights = nn.Parameter(torch.zeros(len(centers), 1 + self.input_dim))

    @property
    def trainable_params(self) -> int:
        return int(self.global_weights.numel() + self.local_weights.numel())

    @property
    def stored_numbers(self) -> int:
        return int(self.trainable_params + self.centers.numel() + self.scales.numel())

    def feature_mats(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_phi = _polynomial_features(z, self.poly_degree)
        if self.centers.numel() == 0:
            local_phi = torch.zeros(len(z), 0, device=z.device, dtype=z.dtype)
            return global_phi, local_phi
        dx = z[:, None, :] - self.centers[None, :, :]
        scales = self.scales[None, :, None].clamp_min(1e-4)
        local = dx / scales
        rbf = torch.exp(-0.5 * (local ** 2).sum(dim=-1))
        local_parts = [rbf]
        for idx in range(self.input_dim):
            local_parts.append(rbf * local[..., idx])
        local_phi = torch.stack(local_parts, dim=-1)
        return global_phi, local_phi.reshape(len(z), -1)

    def decompose(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_phi, local_phi = self.feature_mats(z)
        global_logits = (global_phi @ self.global_weights).unsqueeze(1)
        if local_phi.shape[1] > 0:
            local_logits = (local_phi @ self.local_weights.reshape(-1)).unsqueeze(1)
        else:
            local_logits = torch.zeros(len(z), 1, device=z.device, dtype=z.dtype)
        return global_logits, local_logits

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        global_logits, local_logits = self.decompose(z)
        return global_logits + local_logits


@torch.no_grad()
def _initialize_from_teacher(
    model: ProjectedRBFHybrid,
    z_train: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    ridge: float = 1e-2,
) -> None:
    global_phi, local_phi = model.feature_mats(z_train)
    design = torch.cat([global_phi, local_phi], dim=1)
    design_cpu = design.detach().cpu()
    target_cpu = teacher_logits.detach().cpu()
    gram = design_cpu.T @ design_cpu
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    rhs = design_cpu.T @ target_cpu
    sol = torch.linalg.solve(gram + ridge * eye, rhs).squeeze(1).to(z_train.device)
    g = model.global_weights.numel()
    model.global_weights.copy_(sol[:g])
    if model.local_weights.numel() > 0:
        model.local_weights.copy_(sol[g:].view_as(model.local_weights))


def _train_linear_model(
    model: ProjectedRBFHybrid,
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    z_test: torch.Tensor,
    y_test: torch.Tensor,
    teacher_logits: torch.Tensor,
    epochs: int,
    lr: float,
    sample_weights: torch.Tensor | None = None,
    distill_weight: float = 0.2,
    weight_decay: float = 1e-4,
) -> dict[str, float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_bce = float("inf")
    best_state = None
    best_test = None
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(z_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train, reduction="none")
        if sample_weights is not None:
            loss = loss * sample_weights
        loss = loss.mean() + distill_weight * F.mse_loss(logits, teacher_logits)
        loss.backward()
        opt.step()
        val = _eval_classifier(model, z_val, y_val)
        if val["bce"] < best_bce:
            best_bce = float(val["bce"])
            best_state = copy.deepcopy(model.state_dict())
            best_test = _eval_classifier(model, z_test, y_test)

    if best_state is not None:
        model.load_state_dict(best_state)
    val = _eval_classifier(model, z_val, y_val)
    test = best_test or _eval_classifier(model, z_test, y_test)
    return {
        "val_bce": float(val["bce"]),
        "val_accuracy": float(val["accuracy"]),
        "test_bce": float(test["bce"]),
        "test_accuracy": float(test["accuracy"]),
        "test_margin": float(test["mean_margin"]),
        "params": float(model.trainable_params),
        "stored_numbers": float(model.stored_numbers),
    }


@torch.no_grad()
def _calibrate_residual_scale(
    model: ProjectedRBFHybrid,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    z_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict[str, float]:
    global_val, local_val = model.decompose(z_val)
    global_test, local_test = model.decompose(z_test)

    alpha = torch.tensor([1.0], device=z_val.device, dtype=torch.float32, requires_grad=True)
    bias = torch.tensor([0.0], device=z_val.device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([alpha, bias], max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        logits = global_val + alpha * local_val + bias
        loss = F.binary_cross_entropy_with_logits(logits, y_val)
        loss.backward()
        return loss

    opt.step(closure)
    val_logits = global_val + alpha.detach() * local_val + bias.detach()
    test_logits = global_test + alpha.detach() * local_test + bias.detach()
    val_bce = F.binary_cross_entropy_with_logits(val_logits, y_val).item()
    test_bce = F.binary_cross_entropy_with_logits(test_logits, y_test).item()
    val_probs = torch.sigmoid(val_logits)
    test_probs = torch.sigmoid(test_logits)
    val_acc = ((val_probs > 0.5).float() == y_val).float().mean().item()
    test_acc = ((test_probs > 0.5).float() == y_test).float().mean().item()
    return {
        "alpha": float(alpha.detach().item()),
        "bias": float(bias.detach().item()),
        "val_bce": float(val_bce),
        "val_accuracy": float(val_acc),
        "test_bce": float(test_bce),
        "test_accuracy": float(test_acc),
        "test_margin": float(test_logits.abs().mean().item()),
    }


def run_single(
    seed: int,
    observed_dim: int,
    *,
    dataset: str,
    projection_method: str,
    complexity_mode: str,
    projection_rank: int,
    poly_degree: int,
    source_hidden: int,
    source_depth: int,
    source_epochs: int,
    source_lr: float,
    small_hidden: int,
    small_depth: int,
    small_epochs: int,
    small_lr: float,
    target_budget: int,
    atom_quantile: float,
    fine_tune_epochs: int,
    fine_tune_lr: float,
) -> dict[str, object]:
    cfg = HighDReluCompressionConfig(
        dataset=dataset,
        observed_dims=(observed_dim,),
        n_samples=2400,
        source_hidden=source_hidden,
        source_depth=source_depth,
        source_epochs=source_epochs,
        baseline_epochs=small_epochs,
        cleanup_epochs=fine_tune_epochs,
        device="auto",
        seed=seed,
    )
    _set_seed(seed)
    device = _device_for(cfg)

    benchmark = make_embedded_classification_benchmark(
        dataset=dataset,
        observed_dim=observed_dim,
        n_samples=cfg.n_samples,
        random_state=seed,
    )
    effective_rank = int(projection_rank or benchmark.intrinsic_dim)
    X_train_high = _tensor(benchmark.train.X_high, device)
    y_train = _targets(benchmark.train.y, device)
    X_val_high = _tensor(benchmark.val.X_high, device)
    y_val = _targets(benchmark.val.y, device)
    X_test_high = _tensor(benchmark.test.X_high, device)
    y_test = _targets(benchmark.test.y, device)

    source = _make_classifier(observed_dim, source_hidden, source_depth).to(device)
    source_metrics = _fit_classifier_best(
        source,
        X_train_high,
        y_train,
        X_val=X_val_high,
        y_val=y_val,
        X_test=X_test_high,
        y_test=y_test,
        epochs=source_epochs,
        lr=source_lr,
    )

    ambient_small = _make_classifier(observed_dim, small_hidden, small_depth).to(device)
    ambient_small_metrics = _fit_classifier_best(
        ambient_small,
        X_train_high,
        y_train,
        X_val=X_val_high,
        y_val=y_val,
        X_test=X_test_high,
        y_test=y_test,
        epochs=small_epochs,
        lr=small_lr,
    )

    projection_P, projection_mean = _projection_from_teacher(
        source,
        benchmark.train.X_high,
        method=projection_method,
        rank=effective_rank,
        device=device,
    )
    projection_alignment = _subspace_alignment_score(
        np.asarray(benchmark.embedding.W, dtype=np.float32),
        projection_P,
    )
    if complexity_mode == "teacher_mixed":
        z_train_np, complexity = _teacher_complexity(
            source,
            benchmark.train.X_high,
            projection_P=projection_P,
            projection_mean=projection_mean,
            device=device,
        )
        z_test_np, test_complexity = _teacher_complexity(
            source,
            benchmark.test.X_high,
            projection_P=projection_P,
            projection_mean=projection_mean,
            device=device,
        )
        z_val_np = ((benchmark.val.X_high - projection_mean) @ projection_P).astype(np.float32)
    elif complexity_mode == "projected_field":
        z_train_np = ((benchmark.train.X_high - projection_mean) @ projection_P).astype(np.float32)
        z_val_np = ((benchmark.val.X_high - projection_mean) @ projection_P).astype(np.float32)
        z_test_np = ((benchmark.test.X_high - projection_mean) @ projection_P).astype(np.float32)

        train_logits_np = (
            source(X_train_high).detach().squeeze(1).cpu().numpy().astype(np.float32)
        )
        poly_coeffs = _fit_polynomial_reference(z_train_np, train_logits_np, poly_degree)
        z_train_np, complexity = _projected_field_complexity(
            source,
            benchmark.train.X_high,
            projection_P=projection_P,
            projection_mean=projection_mean,
            poly_coeffs=poly_coeffs,
            poly_degree=poly_degree,
            device=device,
        )
        z_test_np, test_complexity = _projected_field_complexity(
            source,
            benchmark.test.X_high,
            projection_P=projection_P,
            projection_mean=projection_mean,
            poly_coeffs=poly_coeffs,
            poly_degree=poly_degree,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported complexity mode {complexity_mode!r}")

    z_train = _tensor(z_train_np, device)
    z_val = _tensor(z_val_np, device)
    z_test = _tensor(z_test_np, device)
    teacher_logits = source(X_train_high).detach()
    weights = torch.tensor(1.0 + 0.75 * _normalize(complexity), dtype=torch.float32, device=device).unsqueeze(1)

    global_feature_count = len(_monomial_terms(effective_rank, poly_degree))
    atom_budget = max(target_budget - global_feature_count, 0)
    atom_width = 1 + effective_rank
    n_atoms = max(1, min(192, atom_budget // max(atom_width, 1)))
    centers, scales = _build_atom_specs(
        z_train_np,
        complexity,
        n_atoms=n_atoms,
        quantile=atom_quantile,
        seed=seed,
    )
    model = ProjectedRBFHybrid(effective_rank, poly_degree, centers, scales).to(device)
    _initialize_from_teacher(model, z_train, teacher_logits)
    hybrid_metrics = _train_linear_model(
        model,
        z_train,
        y_train,
        z_val=z_val,
        y_val=y_val,
        z_test=z_test,
        y_test=y_test,
        teacher_logits=teacher_logits,
        epochs=fine_tune_epochs,
        lr=fine_tune_lr,
        sample_weights=weights,
        distill_weight=0.20,
        weight_decay=5e-4,
    )
    calibration = _calibrate_residual_scale(model, z_val, y_val, z_test, y_test)

    projected_same_budget, projected_same_budget_meta = _fit_mlp_near_budget(
        effective_rank,
        int(model.trainable_params),
    )
    projected_same_budget = projected_same_budget.to(device)
    projected_same_budget_metrics = _fit_classifier_best(
        projected_same_budget,
        z_train,
        y_train,
        X_val=z_val,
        y_val=y_val,
        X_test=z_test,
        y_test=y_test,
        epochs=fine_tune_epochs,
        lr=min(fine_tune_lr, 3e-3),
        teacher_logits=teacher_logits,
        distill_weight=0.15,
        sample_weights=weights,
    )
    projected_same_budget_metrics.update(projected_same_budget_meta)

    ambient_same_budget, ambient_same_budget_meta = _fit_mlp_near_budget(
        observed_dim,
        int(model.trainable_params),
    )
    ambient_same_budget = ambient_same_budget.to(device)
    ambient_same_budget_metrics = _fit_classifier_best(
        ambient_same_budget,
        X_train_high,
        y_train,
        X_val=X_val_high,
        y_val=y_val,
        X_test=X_test_high,
        y_test=y_test,
        epochs=fine_tune_epochs,
        lr=min(fine_tune_lr, 3e-3),
        teacher_logits=teacher_logits,
        distill_weight=0.15,
        sample_weights=weights,
    )
    ambient_same_budget_metrics.update(ambient_same_budget_meta)

    global_only = ProjectedRBFHybrid(
        effective_rank,
        poly_degree,
        np.zeros((0, effective_rank), dtype=np.float32),
        np.zeros(0, dtype=np.float32),
    ).to(device)
    _initialize_from_teacher(global_only, z_train, teacher_logits)
    global_only_metrics = _train_linear_model(
        global_only,
        z_train,
        y_train,
        z_val=z_val,
        y_val=y_val,
        z_test=z_test,
        y_test=y_test,
        teacher_logits=teacher_logits,
        epochs=fine_tune_epochs,
        lr=fine_tune_lr,
        sample_weights=weights,
        distill_weight=0.15,
        weight_decay=1e-4,
    )

    with torch.no_grad():
        source_logits_test = source(X_test_high).squeeze(1).detach().cpu().numpy().astype(np.float32)
        ambient_logits_test = ambient_small(X_test_high).squeeze(1).detach().cpu().numpy().astype(np.float32)
        ambient_same_budget_logits_test = ambient_same_budget(X_test_high).squeeze(1).detach().cpu().numpy().astype(np.float32)
        global_test_t, local_test_t = model.decompose(z_test)
        hybrid_logits_test = (
            global_test_t + calibration["alpha"] * local_test_t + calibration["bias"]
        ).squeeze(1).detach().cpu().numpy().astype(np.float32)
        global_logits_test = global_only(z_test).squeeze(1).detach().cpu().numpy().astype(np.float32)
        projected_same_budget_logits_test = projected_same_budget(z_test).squeeze(1).detach().cpu().numpy().astype(np.float32)
    bucket_y = benchmark.test.y.astype(np.float32)

    return {
        "source": source_metrics,
        "ambient_small": ambient_small_metrics,
        "projected_same_budget_mlp": projected_same_budget_metrics,
        "ambient_nearest_budget_mlp": ambient_same_budget_metrics,
        "projected_global_poly3": global_only_metrics,
        "hybrid_local_rbf": {
            **hybrid_metrics,
            "calibrated_test_bce": float(calibration["test_bce"]),
            "calibrated_test_accuracy": float(calibration["test_accuracy"]),
            "calibrated_test_margin": float(calibration["test_margin"]),
            "calibration_alpha": float(calibration["alpha"]),
            "calibration_bias": float(calibration["bias"]),
            "n_atoms": float(len(centers)),
            "target_budget": float(target_budget),
        },
        "dataset": dataset,
        "projection_rank": float(effective_rank),
        "poly_degree": float(poly_degree),
        "projection_method": projection_method,
        "complexity_mode": complexity_mode,
        "projection_alignment": float(projection_alignment),
        "complexity_summary": {
            "mean": float(np.mean(complexity)),
            "max": float(np.max(complexity)),
            "q80": float(np.quantile(complexity, 0.8)),
            "q95": float(np.quantile(complexity, 0.95)),
        },
        "teacher_complexity_buckets": {
            "source": _bucket_metrics(source_logits_test, test_complexity, bucket_y),
            "ambient_small": _bucket_metrics(ambient_logits_test, test_complexity, bucket_y),
            "ambient_nearest_budget_mlp": _bucket_metrics(
                ambient_same_budget_logits_test,
                test_complexity,
                bucket_y,
            ),
            "hybrid_local_rbf": _bucket_metrics(hybrid_logits_test, test_complexity, bucket_y),
            "projected_global_poly3": _bucket_metrics(global_logits_test, test_complexity, bucket_y),
            "projected_same_budget_mlp": _bucket_metrics(
                projected_same_budget_logits_test,
                test_complexity,
                bucket_y,
            ),
        },
    }


def _metric_accuracy(row: dict[str, object], key: str) -> float:
    return float(row[key]["test_accuracy"])


def _summarize(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([_metric_accuracy(row, key) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Projected local RBF hybrid on embedded low-D tasks")
    parser.add_argument(
        "--dataset",
        type=str,
        default="rough_spirals",
        choices=["rough_spirals", "rough_hypershell_3d", "rough_hypershell_5d"],
    )
    parser.add_argument("--observed-dims", type=str, default="100,1000")
    parser.add_argument("--projection-methods", type=str, default="manifold_pca")
    parser.add_argument("--complexity-mode", type=str, default="teacher_mixed")
    parser.add_argument("--projection-rank", type=int, default=0)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument("--source-hidden", type=int, default=96)
    parser.add_argument("--source-depth", type=int, default=6)
    parser.add_argument("--source-epochs", type=int, default=1000)
    parser.add_argument("--source-lr", type=float, default=1e-3)
    parser.add_argument("--small-hidden", type=int, default=32)
    parser.add_argument("--small-depth", type=int, default=4)
    parser.add_argument("--small-epochs", type=int, default=1000)
    parser.add_argument("--small-lr", type=float, default=5e-4)
    parser.add_argument("--target-budget", type=int, default=640)
    parser.add_argument("--atom-quantile", type=float, default=0.8)
    parser.add_argument("--fine-tune-epochs", type=int, default=1000)
    parser.add_argument("--fine-tune-lr", type=float, default=5e-3)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dims = [int(part) for part in args.observed_dims.split(",") if part]
    methods = [part.strip() for part in args.projection_methods.split(",") if part.strip()]
    seeds = [int(part) for part in args.seeds.split(",") if part]
    started = time.perf_counter()
    out: dict[str, object] = {
        "experiment": "projected_local_rbf_hybrid",
        "dataset": args.dataset,
        "methods": methods,
        "complexity_mode": args.complexity_mode,
        "projection_rank": float(args.projection_rank),
        "poly_degree": float(args.poly_degree),
        "results": {},
    }

    for method in methods:
        method_rows: dict[str, list[dict[str, object]]] = {str(dim): [] for dim in dims}
        for dim in dims:
            for seed in seeds:
                print(f"=== method {method} dim {dim} seed {seed} start ===", flush=True)
                row = run_single(
                    seed,
                    dim,
                    dataset=args.dataset,
                    projection_method=method,
                    complexity_mode=args.complexity_mode,
                    projection_rank=args.projection_rank,
                    poly_degree=args.poly_degree,
                    source_hidden=args.source_hidden,
                    source_depth=args.source_depth,
                    source_epochs=args.source_epochs,
                    source_lr=args.source_lr,
                    small_hidden=args.small_hidden,
                    small_depth=args.small_depth,
                    small_epochs=args.small_epochs,
                    small_lr=args.small_lr,
                    target_budget=args.target_budget,
                    atom_quantile=args.atom_quantile,
                    fine_tune_epochs=args.fine_tune_epochs,
                    fine_tune_lr=args.fine_tune_lr,
                )
                method_rows[str(dim)].append(row)
                print(f"=== method {method} dim {dim} seed {seed} done ===", flush=True)
        out["results"][method] = {}
        for dim, rows in method_rows.items():
            out["results"][method][dim] = {
                "rows": rows,
                "mean_source_acc": _summarize(rows, "source"),
                "mean_ambient_small_acc": _summarize(rows, "ambient_small"),
                "mean_projected_same_budget_mlp_acc": _summarize(rows, "projected_same_budget_mlp"),
                "mean_ambient_nearest_budget_mlp_acc": _summarize(rows, "ambient_nearest_budget_mlp"),
                "mean_projected_global_poly3_acc": _summarize(rows, "projected_global_poly3"),
                "mean_hybrid_acc": _summarize(rows, "hybrid_local_rbf"),
                "mean_hybrid_vs_small": float(
                    np.mean(
                        [
                            _metric_accuracy(row, "hybrid_local_rbf")
                            - _metric_accuracy(row, "ambient_small")
                            for row in rows
                        ]
                    )
                ),
                "mean_hybrid_vs_projected_same_budget_mlp": float(
                    np.mean(
                        [
                            _metric_accuracy(row, "hybrid_local_rbf")
                            - _metric_accuracy(row, "projected_same_budget_mlp")
                            for row in rows
                        ]
                    )
                ),
                "mean_hybrid_vs_ambient_nearest_budget_mlp": float(
                    np.mean(
                        [
                            _metric_accuracy(row, "hybrid_local_rbf")
                            - _metric_accuracy(row, "ambient_nearest_budget_mlp")
                            for row in rows
                        ]
                    )
                ),
                "mean_hybrid_vs_global_poly3": float(
                    np.mean(
                        [
                            _metric_accuracy(row, "hybrid_local_rbf")
                            - _metric_accuracy(row, "projected_global_poly3")
                            for row in rows
                        ]
                    )
                ),
                "mean_projection_alignment": float(
                    np.mean([float(row["projection_alignment"]) for row in rows])
                ),
                "mean_hybrid_params": float(
                    np.mean([float(row["hybrid_local_rbf"]["params"]) for row in rows])
                ),
                "mean_hybrid_stored_numbers": float(
                    np.mean([float(row["hybrid_local_rbf"]["stored_numbers"]) for row in rows])
                ),
                "mean_n_atoms": float(
                    np.mean([float(row["hybrid_local_rbf"]["n_atoms"]) for row in rows])
                ),
                "mean_projected_same_budget_param_gap": float(
                    np.mean([float(row["projected_same_budget_mlp"]["param_gap"]) for row in rows])
                ),
                "mean_ambient_nearest_budget_param_gap": float(
                    np.mean([float(row["ambient_nearest_budget_mlp"]["param_gap"]) for row in rows])
                ),
            }

    out["wall_time_seconds"] = float(time.perf_counter() - started)
    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
