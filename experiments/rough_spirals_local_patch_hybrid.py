#!/usr/bin/env python3
"""Projected local-patch hybrid compression on rough spirals."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from aleph.datasets import make_embedded_classification_benchmark
from aleph.differential import jacobian_variation
from aleph.models import MLP
from aleph.relu_geometric_compression import (
    _device_for,
    _eval_classifier,
    _forward_hidden,
    _make_classifier,
    _mlp_param_count,
    _set_seed,
    _targets,
    _tensor,
    HighDReluCompressionConfig,
)


@dataclass(frozen=True)
class PatchSpec:
    center: np.ndarray
    scale: float
    mean_complexity: float
    support_size: int


class _ProjectedTeacherView(nn.Module):
    def __init__(self, teacher: nn.Module, projection_P: np.ndarray, projection_mean: np.ndarray):
        super().__init__()
        self.teacher = teacher
        self.register_buffer("P", torch.tensor(projection_P, dtype=torch.float32))
        self.register_buffer("mean", torch.tensor(projection_mean, dtype=torch.float32))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.teacher(self.mean + z @ self.P.T)


class LocalFourierPatchHybrid(nn.Module):
    def __init__(
        self,
        *,
        centers: np.ndarray,
        scales: np.ndarray,
        frequencies: tuple[float, ...],
        global_hidden: int,
        global_depth: int,
    ):
        super().__init__()
        self.global_model = MLP(input_size=2, hidden_size=global_hidden, output_size=1, depth=global_depth)
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.float32))
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("frequencies", torch.tensor(frequencies, dtype=torch.float32))

        basis_dim = 3 + 4 * len(frequencies)
        self.patch_coeffs = nn.Parameter(torch.zeros(len(centers), basis_dim))
        self.bias = nn.Parameter(torch.zeros(1))

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def patch_basis(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = z[:, None, :] - self.centers[None, :, :]
        scales = self.scales[None, :, None].clamp_min(1e-4)
        local = dx / scales
        gate = torch.exp(-0.5 * (local ** 2).sum(dim=-1))

        x = local[..., 0]
        y = local[..., 1]
        parts = [torch.ones_like(x), x, y]
        for freq in self.frequencies:
            parts.extend(
                [
                    torch.sin(freq * x),
                    torch.cos(freq * x),
                    torch.sin(freq * y),
                    torch.cos(freq * y),
                ]
            )
        basis = torch.stack(parts, dim=-1)
        return gate, basis

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.global_model(z)
        if self.centers.numel() == 0:
            return logits + self.bias
        gate, basis = self.patch_basis(z)
        local = (basis * self.patch_coeffs.unsqueeze(0)).sum(dim=-1)
        correction = (gate * local).sum(dim=1, keepdim=True)
        return logits + correction + self.bias


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    lo = float(np.percentile(x, 5))
    hi = float(np.percentile(x, 95))
    if hi <= lo + 1e-8:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _fit_tiny_relu_under_budget(input_dim: int, target_budget: int) -> nn.Module:
    candidates: list[tuple[int, int, int]] = []
    for depth in (1, 2, 3):
        for hidden in range(1, 65):
            params = _mlp_param_count(input_dim, hidden, depth)
            if params <= target_budget:
                candidates.append((params, hidden, depth))
    if candidates:
        _, hidden, depth = max(candidates, key=lambda item: (item[0], -item[1]))
        return _make_classifier(input_dim, hidden, depth)
    return nn.Linear(input_dim, 1)


def _subspace_alignment_score(embedding_W: np.ndarray, projection_P: np.ndarray) -> float:
    q_true, _ = np.linalg.qr(embedding_W.T)
    q_proj, _ = np.linalg.qr(projection_P)
    svals = np.linalg.svd(q_true.T @ q_proj, compute_uv=False)
    return float(np.mean(svals[: min(len(svals), projection_P.shape[1])]))


def _projection_from_teacher(
    teacher: nn.Module,
    X_high: np.ndarray,
    *,
    method: str,
    rank: int = 2,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    mean = X_high.mean(axis=0).astype(np.float32)
    centered = (X_high - mean).astype(np.float32)
    if method == "manifold_pca":
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    elif method == "jacobian_svd":
        x = _tensor(X_high, device)
        grads = []
        for start in range(0, len(x), 256):
            chunk = x[start:start + 256].detach().requires_grad_(True)
            out = teacher(chunk).sum()
            grad = torch.autograd.grad(out, chunk, create_graph=False)[0]
            grads.append(grad.detach().cpu().numpy())
        _, _, vt = np.linalg.svd(np.concatenate(grads, axis=0).astype(np.float32), full_matrices=False)
    else:
        raise ValueError(f"Unsupported projection method {method!r}")
    basis = vt[:rank].T.astype(np.float32)
    q, _ = np.linalg.qr(basis)
    return q[:, :rank].astype(np.float32), mean


def _project_points(X_high: np.ndarray, projection_P: np.ndarray, projection_mean: np.ndarray) -> np.ndarray:
    return ((X_high - projection_mean) @ projection_P).astype(np.float32)


def _activation_churn(
    teacher: nn.Module,
    X_high: np.ndarray,
    projected_points: np.ndarray,
    *,
    device: torch.device,
    knn_k: int = 12,
) -> np.ndarray:
    x = _tensor(X_high, device)
    _, masks, _, _ = _forward_hidden(teacher, x)
    codes = torch.cat([m.float() for m in masks], dim=1).detach().cpu().numpy()
    if len(projected_points) <= 1 or codes.shape[1] == 0:
        return np.zeros(len(projected_points), dtype=np.float32)
    k = min(knn_k + 1, len(projected_points))
    neigh = NearestNeighbors(n_neighbors=k).fit(projected_points).kneighbors(return_distance=False)[:, 1:]
    churn = np.abs(codes[:, None, :] - codes[neigh]).mean(axis=2).mean(axis=1)
    return churn.astype(np.float32)


def _projected_boundary_density(
    teacher: nn.Module,
    projected_points: np.ndarray,
    projection_P: np.ndarray,
    projection_mean: np.ndarray,
    *,
    device: torch.device,
    eps: float = 0.04,
    n_perturbations: int = 8,
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


def _teacher_complexity(
    teacher: nn.Module,
    X_high: np.ndarray,
    *,
    projection_P: np.ndarray,
    projection_mean: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    projected = _project_points(X_high, projection_P, projection_mean)
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
    churn = _activation_churn(teacher, X_high, projected, device=device, knn_k=12)
    boundary = _projected_boundary_density(
        teacher,
        projected,
        projection_P,
        projection_mean,
        device=device,
        eps=0.04,
        n_perturbations=8,
    )
    complexity = (
        0.35 * _normalize(jac_var)
        + 0.25 * _normalize(churn)
        + 0.20 * _normalize(boundary)
        + 0.20 * _normalize(margin_inv)
    ).astype(np.float32)
    return projected, complexity


def _build_patch_specs(
    projected: np.ndarray,
    complexity: np.ndarray,
    *,
    patch_count: int,
    quantile: float,
    min_points: int = 60,
    seed: int,
) -> list[PatchSpec]:
    threshold = float(np.quantile(complexity, quantile))
    hard_mask = complexity >= threshold
    if int(hard_mask.sum()) < min_points:
        topk = min(max(min_points, patch_count * 20), len(projected))
        keep = np.argsort(complexity)[-topk:]
        hard_mask = np.zeros(len(projected), dtype=bool)
        hard_mask[keep] = True
    hard_points = projected[hard_mask]
    hard_weights = complexity[hard_mask]
    if len(hard_points) == 0:
        return []

    n_clusters = min(patch_count, len(hard_points))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(
        hard_points,
        sample_weight=hard_weights,
    )

    specs: list[PatchSpec] = []
    for cluster in range(n_clusters):
        mask = labels == cluster
        pts = hard_points[mask]
        weights = hard_weights[mask]
        if len(pts) == 0:
            continue
        center = np.average(pts, axis=0, weights=weights).astype(np.float32)
        sqdist = np.sum((pts - center) ** 2, axis=1)
        scale = float(np.sqrt(np.average(sqdist, weights=weights) + 1e-4))
        scale = float(np.clip(scale * 1.35, 0.08, 1.25))
        specs.append(
            PatchSpec(
                center=center,
                scale=scale,
                mean_complexity=float(np.average(weights)),
                support_size=int(len(pts)),
            )
        )
    specs.sort(key=lambda item: (item.mean_complexity, item.support_size), reverse=True)
    return specs


def _fit_classifier_best(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int,
    lr: float,
    teacher_logits: torch.Tensor | None = None,
    distill_weight: float = 0.0,
    sample_weights: torch.Tensor | None = None,
) -> dict[str, object]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_bce = float("inf")
    best_state = None
    best_test = None
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train, reduction="none")
        if sample_weights is not None:
            loss = loss * sample_weights
        loss = loss.mean()
        if teacher_logits is not None and distill_weight > 0:
            loss = loss + distill_weight * F.mse_loss(logits, teacher_logits)
        loss.backward()
        opt.step()

        val = _eval_classifier(model, X_val, y_val)
        if val["bce"] < best_bce:
            best_bce = float(val["bce"])
            best_state = copy.deepcopy(model.state_dict())
            best_test = _eval_classifier(model, X_test, y_test)

    if best_state is not None:
        model.load_state_dict(best_state)
    val = _eval_classifier(model, X_val, y_val)
    test = best_test or _eval_classifier(model, X_test, y_test)
    return {
        "val_bce": float(val["bce"]),
        "val_accuracy": float(val["accuracy"]),
        "test_bce": float(test["bce"]),
        "test_accuracy": float(test["accuracy"]),
        "test_margin": float(test["mean_margin"]),
        "params": float(sum(p.numel() for p in model.parameters())),
    }


@torch.no_grad()
def _fit_patch_coefficients_from_teacher(
    hybrid: LocalFourierPatchHybrid,
    z_train: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> None:
    if hybrid.centers.numel() == 0:
        hybrid.bias.zero_()
        return
    global_logits = hybrid.global_model(z_train).squeeze(1)
    residual = (teacher_logits.squeeze(1) - global_logits).detach()
    gate, basis = hybrid.patch_basis(z_train)
    design = (gate.unsqueeze(-1) * basis).reshape(len(z_train), -1)
    design = torch.cat([design, torch.ones(len(z_train), 1, device=z_train.device)], dim=1)
    sol = torch.linalg.lstsq(
        design.detach().cpu(),
        residual.unsqueeze(1).detach().cpu(),
    ).solution.squeeze(1).to(z_train.device)
    coeff_count = hybrid.patch_coeffs.numel()
    hybrid.patch_coeffs.copy_(sol[:coeff_count].view_as(hybrid.patch_coeffs))
    hybrid.bias.copy_(sol[coeff_count:coeff_count + 1])


def run_single(
    seed: int,
    observed_dim: int,
    *,
    projection_method: str,
    source_hidden: int,
    source_depth: int,
    source_epochs: int,
    source_lr: float,
    small_hidden: int,
    small_depth: int,
    small_epochs: int,
    small_lr: float,
    global_hidden: int,
    global_depth: int,
    patch_count: int,
    patch_quantile: float,
    fine_tune_epochs: int,
    fine_tune_lr: float,
) -> dict[str, object]:
    cfg = HighDReluCompressionConfig(
        dataset="rough_spirals",
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
        dataset="rough_spirals",
        observed_dim=observed_dim,
        n_samples=cfg.n_samples,
        random_state=seed,
    )
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
        device=device,
    )
    projection_alignment = _subspace_alignment_score(
        np.asarray(benchmark.embedding.W, dtype=np.float32),
        projection_P,
    )
    z_train_np, complexity = _teacher_complexity(
        source,
        benchmark.train.X_high,
        projection_P=projection_P,
        projection_mean=projection_mean,
        device=device,
    )
    z_val_np = _project_points(benchmark.val.X_high, projection_P, projection_mean)
    z_test_np = _project_points(benchmark.test.X_high, projection_P, projection_mean)

    z_train = _tensor(z_train_np, device)
    z_val = _tensor(z_val_np, device)
    z_test = _tensor(z_test_np, device)
    teacher_logits = source(X_train_high).detach()
    sample_weights = torch.tensor(1.0 + 0.75 * _normalize(complexity), dtype=torch.float32, device=device).unsqueeze(1)
    smooth_weights = torch.tensor(
        1.0 - 0.55 * _normalize(complexity),
        dtype=torch.float32,
        device=device,
    ).clamp_min_(0.2).unsqueeze(1)

    patch_specs = _build_patch_specs(
        z_train_np,
        complexity,
        patch_count=patch_count,
        quantile=patch_quantile,
        seed=seed,
    )
    centers = np.stack([spec.center for spec in patch_specs], axis=0) if patch_specs else np.zeros((0, 2), dtype=np.float32)
    scales = np.asarray([spec.scale for spec in patch_specs], dtype=np.float32) if patch_specs else np.zeros(0, dtype=np.float32)

    hybrid = LocalFourierPatchHybrid(
        centers=centers,
        scales=scales,
        frequencies=(1.0, 2.0, 4.0),
        global_hidden=global_hidden,
        global_depth=global_depth,
    ).to(device)
    global_pretrain = _fit_classifier_best(
        hybrid.global_model,
        z_train,
        y_train,
        X_val=z_val,
        y_val=y_val,
        X_test=z_test,
        y_test=y_test,
        epochs=max(250, fine_tune_epochs // 2),
        lr=2e-3,
        teacher_logits=None,
        distill_weight=0.0,
        sample_weights=smooth_weights,
    )
    _fit_patch_coefficients_from_teacher(hybrid, z_train, teacher_logits)
    hybrid_metrics = _fit_classifier_best(
        hybrid,
        z_train,
        y_train,
        X_val=z_val,
        y_val=y_val,
        X_test=z_test,
        y_test=y_test,
        epochs=fine_tune_epochs,
        lr=fine_tune_lr,
        teacher_logits=teacher_logits,
        distill_weight=0.25,
        sample_weights=sample_weights,
    )

    projected_tiny = _fit_tiny_relu_under_budget(2, int(hybrid.trainable_params)).to(device)
    projected_tiny_metrics = _fit_classifier_best(
        projected_tiny,
        z_train,
        y_train,
        X_val=z_val,
        y_val=y_val,
        X_test=z_test,
        y_test=y_test,
        epochs=fine_tune_epochs,
        lr=fine_tune_lr,
        teacher_logits=teacher_logits,
        distill_weight=0.20,
        sample_weights=sample_weights,
    )

    return {
        "source": source_metrics,
        "ambient_small": ambient_small_metrics,
        "projected_global_pretrain": global_pretrain,
        "projected_tiny_relu": projected_tiny_metrics,
        "hybrid_local_fourier": {
            **hybrid_metrics,
            "patch_count": float(len(patch_specs)),
            "patch_support": [int(spec.support_size) for spec in patch_specs],
            "patch_scales": [float(spec.scale) for spec in patch_specs],
            "patch_mean_complexity": [float(spec.mean_complexity) for spec in patch_specs],
        },
        "projection_method": projection_method,
        "projection_alignment": float(projection_alignment),
        "complexity_summary": {
            "mean": float(np.mean(complexity)),
            "max": float(np.max(complexity)),
            "q80": float(np.quantile(complexity, 0.8)),
            "q95": float(np.quantile(complexity, 0.95)),
        },
    }


def _summarize(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]["test_accuracy"]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Projected local-patch hybrid on rough spirals")
    parser.add_argument("--observed-dims", type=str, default="100,1000")
    parser.add_argument("--projection-methods", type=str, default="manifold_pca")
    parser.add_argument("--source-hidden", type=int, default=96)
    parser.add_argument("--source-depth", type=int, default=6)
    parser.add_argument("--source-epochs", type=int, default=1000)
    parser.add_argument("--source-lr", type=float, default=1e-3)
    parser.add_argument("--small-hidden", type=int, default=32)
    parser.add_argument("--small-depth", type=int, default=4)
    parser.add_argument("--small-epochs", type=int, default=1000)
    parser.add_argument("--small-lr", type=float, default=5e-4)
    parser.add_argument("--global-hidden", type=int, default=8)
    parser.add_argument("--global-depth", type=int, default=2)
    parser.add_argument("--patch-count", type=int, default=6)
    parser.add_argument("--patch-quantile", type=float, default=0.8)
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
        "experiment": "rough_spirals_local_patch_hybrid",
        "methods": methods,
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
                    projection_method=method,
                    source_hidden=args.source_hidden,
                    source_depth=args.source_depth,
                    source_epochs=args.source_epochs,
                    source_lr=args.source_lr,
                    small_hidden=args.small_hidden,
                    small_depth=args.small_depth,
                    small_epochs=args.small_epochs,
                    small_lr=args.small_lr,
                    global_hidden=args.global_hidden,
                    global_depth=args.global_depth,
                    patch_count=args.patch_count,
                    patch_quantile=args.patch_quantile,
                    fine_tune_epochs=args.fine_tune_epochs,
                    fine_tune_lr=args.fine_tune_lr,
                )
                method_rows[str(dim)].append(row)
                print(f"=== method {method} dim {dim} seed {seed} done ===", flush=True)

                if args.output:
                    partial = {
                        "experiment": "rough_spirals_local_patch_hybrid",
                        "methods": methods,
                        "results": out["results"],
                    }
                    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
                    with open(args.output, "w") as f:
                        json.dump(partial, f, indent=2)

        out["results"][method] = {}
        for dim, rows in method_rows.items():
            out["results"][method][dim] = {
                "rows": rows,
                "mean_source_acc": _summarize(rows, "source"),
                "mean_ambient_small_acc": _summarize(rows, "ambient_small"),
                "mean_projected_tiny_acc": _summarize(rows, "projected_tiny_relu"),
                "mean_hybrid_acc": _summarize(rows, "hybrid_local_fourier"),
                "mean_hybrid_vs_small": float(
                    np.mean(
                        [
                            float(row["hybrid_local_fourier"]["test_accuracy"])
                            - float(row["ambient_small"]["test_accuracy"])
                            for row in rows
                        ]
                    )
                ),
                "mean_hybrid_vs_tiny": float(
                    np.mean(
                        [
                            float(row["hybrid_local_fourier"]["test_accuracy"])
                            - float(row["projected_tiny_relu"]["test_accuracy"])
                            for row in rows
                        ]
                    )
                ),
                "mean_patch_count": float(
                    np.mean([float(row["hybrid_local_fourier"]["patch_count"]) for row in rows])
                ),
                "mean_hybrid_params": float(
                    np.mean([float(row["hybrid_local_fourier"]["params"]) for row in rows])
                ),
                "mean_projection_alignment": float(
                    np.mean([float(row["projection_alignment"]) for row in rows])
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
