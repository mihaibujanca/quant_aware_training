"""High-D-first geometric compression for ReLU MLP classifiers.

This module implements a fresh compression path centered on:
- low-dimensional datasets embedded into high-dimensional observed inputs
- data-active piecewise-affine region analysis
- geometry-guided coarsening into macro-regions
- synthesis of a smaller ReLU MLP

Native 2D runs are kept as diagnostics only. The primary benchmarks are the
embedded settings (for example 2->100 and 2->1000).
"""

from __future__ import annotations

import json
import math
import time

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from aleph.datasets import (
    EmbeddedClassificationBenchmark,
    IdentityEmbedding,
    make_embedded_classification_benchmark,
)
from aleph.differential import gradient, jacobian_variation
from aleph.models import MLP


@dataclass(frozen=True)
class RegionInfo:
    region_id: int
    code: str
    support_size: int
    empirical_mass: float
    predicted_class: int
    mean_margin: float
    affine_weights: np.ndarray
    affine_bias: float
    centroid_low: np.ndarray
    centroid_high: np.ndarray
    mean_curvature: float
    mean_boundary_density: float
    mean_hardness: float
    mean_churn: float
    mean_region_density: float
    mean_jacobian_change: float
    mean_mergeability: float = 0.0


@dataclass(frozen=True)
class RegionAtlas:
    benchmark_name: str
    observed_dim: int
    sample_region_ids: np.ndarray
    sample_activation_codes: tuple[str, ...]
    sample_logits: np.ndarray
    sample_margins: np.ndarray
    sample_curvature: np.ndarray
    sample_boundary_density: np.ndarray
    sample_hardness: np.ndarray
    sample_activation_churn: np.ndarray
    sample_region_density: np.ndarray
    sample_jacobian_change: np.ndarray
    regions: tuple[RegionInfo, ...]
    adjacency: dict[int, tuple[int, ...]]
    correlations: dict[str, float]
    stable_neuron_stats: tuple[dict[str, int], ...]


@dataclass(frozen=True)
class MacroRegionInfo:
    macro_region_id: int
    member_regions: tuple[int, ...]
    support_size: int
    empirical_mass: float
    predicted_class: int
    mean_margin: float
    affine_weights: np.ndarray
    affine_bias: float
    centroid_low: np.ndarray
    centroid_high: np.ndarray
    mean_curvature: float
    mean_boundary_density: float
    mean_hardness: float
    mean_mergeability: float


@dataclass(frozen=True)
class MacroRegionAtlas:
    sample_macro_ids: np.ndarray
    macro_regions: tuple[MacroRegionInfo, ...]
    adjacency: dict[int, tuple[int, ...]]
    merge_edge_count: int


@dataclass(frozen=True)
class CompressionArchitecture:
    input_dim: int
    hidden_size: int
    depth: int
    params: int


@dataclass
class HighDReluCompressionConfig:
    dataset: str = "spirals"
    observed_dims: tuple[int, ...] = (2, 100, 1000)
    n_samples: int = 2400
    noise: float | None = None
    n_turns: int = 3
    val_fraction: float = 0.2
    test_fraction: float = 0.2

    source_hidden: int = 64
    source_depth: int = 4
    source_epochs: int = 800
    source_lr: float = 1e-3

    baseline_epochs: int = 240
    baseline_lr: float = 5e-4
    cleanup_epochs: int = 180
    cleanup_lr: float = 5e-4
    distill_weight: float = 0.6
    geometry_distill_weight: float = 0.35
    geometry_jacobian_weight: float = 0.2

    knn_k: int = 12
    curvature_eps: float = 0.03
    boundary_eps: float = 0.06
    boundary_perturbations: int = 10

    low_complexity_quantile: float = 0.6
    max_affine_gap: float = 0.20
    max_merge_bce_increase: float = 0.03
    min_merge_margin: float = 1.0

    architecture_min_hidden: int = 8
    architecture_max_hidden: int = 32
    device: str = "cpu"
    seed: int = 42
    trace_every_n_steps: int = 0


class EmbeddedLowDimView(nn.Module):
    """Low-dimensional view of a high-dimensional observed model."""

    def __init__(self, model: nn.Module, embedding: Any):
        super().__init__()
        self.model = model
        W = torch.tensor(embedding.W, dtype=torch.float32)
        b = torch.tensor(embedding.b, dtype=torch.float32)
        self.register_buffer("W", W)
        self.register_buffer("b", b)

    def forward(self, x_low: torch.Tensor) -> torch.Tensor:
        return self.model(x_low @ self.W + self.b)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _device_for(cfg: HighDReluCompressionConfig) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg.device)


def _make_classifier(input_dim: int, hidden_size: int, depth: int) -> MLP:
    return MLP(input_size=input_dim, hidden_size=hidden_size, output_size=1, depth=depth)


def _linear_layers(model: nn.Module) -> list[nn.Linear]:
    if not hasattr(model, "layers") or not isinstance(model.layers, nn.Sequential):
        raise ValueError("Expected aleph.models.MLP with .layers Sequential.")
    layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
    if len(layers) < 2:
        raise ValueError("Expected at least one hidden layer and one output layer.")
    return layers


def _hidden_layer_widths(model: nn.Module) -> list[int]:
    return [layer.out_features for layer in _linear_layers(model)[:-1]]


def _mlp_param_count(input_dim: int, hidden_size: int, depth: int, output_dim: int = 1) -> int:
    total = input_dim * hidden_size + hidden_size
    total += max(depth - 1, 0) * (hidden_size * hidden_size + hidden_size)
    total += hidden_size * output_dim + output_dim
    return total


def _tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _targets(y: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)


def _num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _sigmoid_margin(logits: torch.Tensor) -> torch.Tensor:
    return logits.abs().squeeze(-1)


def _weighted_bce(
    logits: torch.Tensor,
    y: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none").squeeze(-1)
    if sample_weights is not None:
        loss = loss * sample_weights
    return loss.mean()


def _train_classifier(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    teacher_logits: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    jacobian_points_low: torch.Tensor | None = None,
    teacher_jacobians_low: torch.Tensor | None = None,
    low_dim_view: EmbeddedLowDimView | None = None,
    distill_weight: float = 0.0,
    jacobian_weight: float = 0.0,
    X_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    X_test: torch.Tensor | None = None,
    y_test: torch.Tensor | None = None,
    trace_every_n_steps: int = 0,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trace: list[dict[str, float]] = []
    for step in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = _weighted_bce(logits, y, sample_weights=sample_weights)
        if teacher_logits is not None and distill_weight > 0:
            loss = loss + distill_weight * F.mse_loss(logits, teacher_logits)
        if (
            jacobian_points_low is not None
            and teacher_jacobians_low is not None
            and low_dim_view is not None
            and jacobian_weight > 0
        ):
            student_jac = gradient(low_dim_view, jacobian_points_low, chunk_size=256)
            loss = loss + jacobian_weight * F.mse_loss(student_jac, teacher_jacobians_low)
        loss.backward()
        optimizer.step()
        if trace_every_n_steps > 0 and ((step + 1) % trace_every_n_steps == 0 or step == 0 or step + 1 == epochs):
            row = {
                "step": float(step + 1),
                "train_loss": float(loss.detach().item()),
            }
            if X_val is not None and y_val is not None:
                val_metrics = _eval_classifier(model, X_val, y_val)
                row["val_bce"] = float(val_metrics["bce"])
                row["val_accuracy"] = float(val_metrics["accuracy"])
            if X_test is not None and y_test is not None:
                test_metrics = _eval_classifier(model, X_test, y_test)
                row["test_bce"] = float(test_metrics["bce"])
                row["test_accuracy"] = float(test_metrics["accuracy"])
            trace.append(row)
    return trace


@torch.no_grad()
def _eval_classifier(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    logits = model(X)
    bce = F.binary_cross_entropy_with_logits(logits, y).item()
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    acc = (preds == y).float().mean().item()
    return {
        "bce": bce,
        "accuracy": acc,
        "mean_margin": _sigmoid_margin(logits).mean().item(),
    }


@torch.no_grad()
def _forward_hidden(
    model: nn.Module, X: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    linears = _linear_layers(model)
    a = X
    masks: list[torch.Tensor] = []
    hiddens: list[torch.Tensor] = []
    for linear in linears[:-1]:
        z = linear(a)
        mask = z > 0
        a = F.relu(z)
        masks.append(mask)
        hiddens.append(a)
    logits = linears[-1](a)
    return logits, masks, hiddens, a


def _pack_activation_codes(masks: list[torch.Tensor]) -> tuple[np.ndarray, tuple[str, ...]]:
    flat = torch.cat([m.to(torch.uint8) for m in masks], dim=1).cpu().numpy()
    packed = np.packbits(flat, axis=1)
    keys = tuple(bytes(row).hex() for row in packed)
    return flat.astype(bool), keys


def _effective_affine_from_mask(model: nn.Module, mask_layers: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    linears = _linear_layers(model)
    input_dim = linears[0].in_features
    A = torch.eye(input_dim, dtype=torch.float32)
    b = torch.zeros(input_dim, dtype=torch.float32)
    for linear, mask in zip(linears[:-1], mask_layers):
        W = linear.weight.detach().cpu()
        bias = linear.bias.detach().cpu()
        A = W @ A
        b = W @ b + bias
        D = torch.diag(mask.float().cpu())
        A = D @ A
        b = D @ b
    W_out = linears[-1].weight.detach().cpu()
    b_out = linears[-1].bias.detach().cpu()
    A = W_out @ A
    b = W_out @ b + b_out
    return A.squeeze(0), b.squeeze(0)


def _fit_affine_map(X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    orig_device = X.device
    X_cpu = X.detach().to("cpu")
    y_cpu = y.detach().to("cpu")
    X_aug = torch.cat([X_cpu, torch.ones(len(X_cpu), 1, device=X_cpu.device)], dim=1)
    sol = torch.linalg.lstsq(X_aug, y_cpu.unsqueeze(1)).solution.squeeze(1)
    return sol[:-1].to(orig_device), sol[-1].to(orig_device)


def _intrinsic_boundary_density(
    model: nn.Module,
    embedding: Any,
    X_low: torch.Tensor,
    *,
    eps: float,
    n_perturbations: int,
    chunk_size: int = 512,
) -> torch.Tensor:
    density = torch.zeros(len(X_low), device=X_low.device)
    W = torch.tensor(embedding.W, dtype=torch.float32, device=X_low.device)
    b = torch.tensor(embedding.b, dtype=torch.float32, device=X_low.device)
    for _ in range(n_perturbations):
        direction = torch.randn_like(X_low)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
        X_pert = X_low + eps * direction
        base_high = X_low @ W + b
        pert_high = X_pert @ W + b
        base_masks = []
        pert_masks = []
        for start in range(0, len(X_low), chunk_size):
            _, base_layer_masks, _, _ = _forward_hidden(model, base_high[start:start + chunk_size])
            _, pert_layer_masks, _, _ = _forward_hidden(model, pert_high[start:start + chunk_size])
            base_masks.append(torch.cat(base_layer_masks, dim=1))
            pert_masks.append(torch.cat(pert_layer_masks, dim=1))
        base_mask = torch.cat(base_masks, dim=0)
        pert_mask = torch.cat(pert_masks, dim=0)
        density += (base_mask != pert_mask).float().sum(dim=1)
    return density / max(n_perturbations, 1)


def _neuron_stability(masks: list[torch.Tensor]) -> tuple[dict[str, int], ...]:
    stats = []
    for layer_mask in masks:
        always_on = int(layer_mask.all(dim=0).sum().item())
        always_off = int((~layer_mask).all(dim=0).sum().item())
        mixed = int(layer_mask.shape[1] - always_on - always_off)
        stats.append(
            {
                "always_on": always_on,
                "always_off": always_off,
                "mixed": mixed,
                "total": int(layer_mask.shape[1]),
            }
        )
    return tuple(stats)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    corr = spearmanr(x, y).correlation
    return float(0.0 if corr is None or np.isnan(corr) else corr)


def build_region_atlas(
    model: nn.Module,
    benchmark: EmbeddedClassificationBenchmark,
    *,
    device: torch.device,
    knn_k: int,
    curvature_eps: float,
    boundary_eps: float,
    boundary_perturbations: int,
) -> RegionAtlas:
    X_high = _tensor(benchmark.train.X_high, device)
    X_low = _tensor(benchmark.train.X_low, device)
    y = _targets(benchmark.train.y, device)

    logits, masks, _, _ = _forward_hidden(model, X_high)
    flat_masks, code_keys = _pack_activation_codes(masks)
    logits_np = logits.squeeze(1).detach().cpu().numpy()
    margins_np = np.abs(logits_np)

    low_dim_view = EmbeddedLowDimView(model, benchmark.embedding).to(device)
    low_dim_view.eval()
    curvature = jacobian_variation(low_dim_view, X_low, eps=curvature_eps, chunk_size=256).detach().cpu().numpy()
    boundary_density = _intrinsic_boundary_density(
        model,
        benchmark.embedding,
        X_low,
        eps=boundary_eps,
        n_perturbations=boundary_perturbations,
    ).detach().cpu().numpy()

    margin_inv = 1.0 / (margins_np + 1e-3)
    hardness = 0.5 * _normalize_array(curvature) + 0.5 * _normalize_array(margin_inv)

    region_id_map: dict[str, int] = {}
    support_by_region: dict[int, list[int]] = defaultdict(list)
    region_masks: dict[int, list[torch.Tensor]] = {}
    sample_region_ids = np.zeros(len(code_keys), dtype=np.int64)
    for idx, code in enumerate(code_keys):
        region_id = region_id_map.setdefault(code, len(region_id_map))
        sample_region_ids[idx] = region_id
        support_by_region[region_id].append(idx)
        if region_id not in region_masks:
            per_layer = []
            cursor = 0
            for layer_mask in masks:
                width = layer_mask.shape[1]
                per_layer.append(layer_mask[idx].detach().cpu())
                cursor += width
            region_masks[region_id] = per_layer

    nbrs = NearestNeighbors(n_neighbors=min(knn_k + 1, len(sample_region_ids))).fit(benchmark.train.X_low)
    knn_indices = nbrs.kneighbors(return_distance=False)[:, 1:]
    sample_churn = np.zeros(len(sample_region_ids), dtype=np.float32)
    sample_region_density = np.zeros(len(sample_region_ids), dtype=np.float32)
    sample_jacobian_change = np.zeros(len(sample_region_ids), dtype=np.float32)
    adjacency_sets: dict[int, set[int]] = defaultdict(set)

    region_affines: dict[int, tuple[np.ndarray, float]] = {}
    for region_id, mask_layers in region_masks.items():
        A, b = _effective_affine_from_mask(model, mask_layers)
        region_affines[region_id] = (A.numpy().astype(np.float32), float(b.item()))

    affine_by_sample = np.stack([region_affines[r][0] for r in sample_region_ids], axis=0)
    for idx, neigh in enumerate(knn_indices):
        neigh_regions = sample_region_ids[neigh]
        sample_churn[idx] = float(np.mean(neigh_regions != sample_region_ids[idx]))
        sample_region_density[idx] = float(len(np.unique(np.append(neigh_regions, sample_region_ids[idx]))))
        neigh_affine = affine_by_sample[neigh]
        sample_jacobian_change[idx] = float(np.mean(np.linalg.norm(neigh_affine - affine_by_sample[idx], axis=1)))
        for neigh_region in neigh_regions:
            if neigh_region != sample_region_ids[idx]:
                adjacency_sets[sample_region_ids[idx]].add(int(neigh_region))
                adjacency_sets[int(neigh_region)].add(int(sample_region_ids[idx]))

    regions = []
    for region_id in range(len(region_id_map)):
        support = np.asarray(support_by_region[region_id], dtype=np.int64)
        affine_w, affine_b = region_affines[region_id]
        mean_logit = float(np.mean(logits_np[support]))
        regions.append(
            RegionInfo(
                region_id=region_id,
                code=next(code for code, rid in region_id_map.items() if rid == region_id),
                support_size=int(len(support)),
                empirical_mass=float(len(support) / len(sample_region_ids)),
                predicted_class=int(mean_logit > 0),
                mean_margin=float(np.mean(margins_np[support])),
                affine_weights=affine_w,
                affine_bias=affine_b,
                centroid_low=np.mean(benchmark.train.X_low[support], axis=0).astype(np.float32),
                centroid_high=np.mean(benchmark.train.X_high[support], axis=0).astype(np.float32),
                mean_curvature=float(np.mean(curvature[support])),
                mean_boundary_density=float(np.mean(boundary_density[support])),
                mean_hardness=float(np.mean(hardness[support])),
                mean_churn=float(np.mean(sample_churn[support])),
                mean_region_density=float(np.mean(sample_region_density[support])),
                mean_jacobian_change=float(np.mean(sample_jacobian_change[support])),
            )
        )

    correlations = {
        "curvature_vs_region_density": _spearman(curvature, sample_region_density),
        "curvature_vs_activation_churn": _spearman(curvature, sample_churn),
        "curvature_vs_jacobian_change": _spearman(curvature, sample_jacobian_change),
        "hardness_vs_region_density": _spearman(hardness, sample_region_density),
        "hardness_vs_activation_churn": _spearman(hardness, sample_churn),
        "hardness_vs_jacobian_change": _spearman(hardness, sample_jacobian_change),
        "curvature_vs_boundary_density": _spearman(curvature, boundary_density),
    }
    adjacency = {rid: tuple(sorted(neigh)) for rid, neigh in adjacency_sets.items()}
    for rid in range(len(region_id_map)):
        adjacency.setdefault(rid, tuple())

    return RegionAtlas(
        benchmark_name=benchmark.name,
        observed_dim=benchmark.observed_dim,
        sample_region_ids=sample_region_ids,
        sample_activation_codes=code_keys,
        sample_logits=logits_np.astype(np.float32),
        sample_margins=margins_np.astype(np.float32),
        sample_curvature=curvature.astype(np.float32),
        sample_boundary_density=boundary_density.astype(np.float32),
        sample_hardness=hardness.astype(np.float32),
        sample_activation_churn=sample_churn.astype(np.float32),
        sample_region_density=sample_region_density.astype(np.float32),
        sample_jacobian_change=sample_jacobian_change.astype(np.float32),
        regions=tuple(regions),
        adjacency=adjacency,
        correlations=correlations,
        stable_neuron_stats=_neuron_stability(masks),
    )


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return False
        self.parent[ry] = rx
        return True


def _normalize_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if math.isclose(lo, hi):
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def coarsen_region_atlas(
    atlas: RegionAtlas,
    benchmark: EmbeddedClassificationBenchmark,
    *,
    max_affine_gap: float,
    max_merge_bce_increase: float,
    min_merge_margin: float,
    low_complexity_quantile: float,
) -> MacroRegionAtlas:
    regions_by_id = {region.region_id: region for region in atlas.regions}
    y = torch.tensor(benchmark.train.y, dtype=torch.float32)
    source_logits = torch.tensor(atlas.sample_logits, dtype=torch.float32)
    X_high = torch.tensor(benchmark.train.X_high, dtype=torch.float32)

    curvature_thresh = float(np.quantile([r.mean_curvature for r in atlas.regions], low_complexity_quantile))
    boundary_thresh = float(
        np.quantile([r.mean_boundary_density for r in atlas.regions], low_complexity_quantile)
    )

    edge_scores: list[tuple[float, int, int]] = []
    region_mergeability = defaultdict(list)
    for region_id, neighbors in atlas.adjacency.items():
        for neigh_id in neighbors:
            if neigh_id <= region_id:
                continue
            left = regions_by_id[region_id]
            right = regions_by_id[neigh_id]
            if left.predicted_class != right.predicted_class:
                continue
            if min(left.mean_margin, right.mean_margin) < min_merge_margin:
                continue
            if max(left.mean_curvature, right.mean_curvature) > curvature_thresh:
                continue
            if max(left.mean_boundary_density, right.mean_boundary_density) > boundary_thresh:
                continue

            affine_gap = np.linalg.norm(left.affine_weights - right.affine_weights)
            affine_gap /= np.linalg.norm(np.concatenate([left.affine_weights, right.affine_weights])) + 1e-6
            if affine_gap > max_affine_gap:
                continue

            support_mask = np.isin(atlas.sample_region_ids, [region_id, neigh_id])
            support_idx = np.flatnonzero(support_mask)
            fit_w, fit_b = _fit_affine_map(X_high[support_idx], source_logits[support_idx])
            merged_logits = X_high[support_idx] @ fit_w + fit_b
            merged_bce = F.binary_cross_entropy_with_logits(merged_logits, y[support_idx]).item()
            source_bce = F.binary_cross_entropy_with_logits(
                source_logits[support_idx], y[support_idx]
            ).item()
            bce_increase = merged_bce - source_bce
            if bce_increase > max_merge_bce_increase:
                continue

            score = float(
                affine_gap
                + 0.35 * (left.mean_hardness + right.mean_hardness)
                + 0.25 * max(bce_increase, 0.0)
            )
            edge_scores.append((score, region_id, neigh_id))
            region_mergeability[region_id].append(score)
            region_mergeability[neigh_id].append(score)

    uf = _UnionFind(len(atlas.regions))
    merge_edge_count = 0
    for _, left, right in sorted(edge_scores):
        if uf.union(left, right):
            merge_edge_count += 1

    root_to_members: dict[int, list[int]] = defaultdict(list)
    for region in atlas.regions:
        root_to_members[uf.find(region.region_id)].append(region.region_id)

    sample_macro_ids = np.asarray([uf.find(int(rid)) for rid in atlas.sample_region_ids], dtype=np.int64)
    macro_id_map = {root: idx for idx, root in enumerate(sorted(root_to_members))}
    sample_macro_ids = np.asarray([macro_id_map[int(root)] for root in sample_macro_ids], dtype=np.int64)

    macro_regions = []
    for root, members in sorted(root_to_members.items(), key=lambda item: macro_id_map[item[0]]):
        macro_id = macro_id_map[root]
        support = np.flatnonzero(np.isin(atlas.sample_region_ids, members))
        fit_w, fit_b = _fit_affine_map(X_high[support], source_logits[support])
        member_objs = [regions_by_id[mid] for mid in members]
        macro_regions.append(
            MacroRegionInfo(
                macro_region_id=macro_id,
                member_regions=tuple(sorted(members)),
                support_size=int(len(support)),
                empirical_mass=float(len(support) / len(atlas.sample_region_ids)),
                predicted_class=int(np.mean(atlas.sample_logits[support]) > 0),
                mean_margin=float(np.mean(atlas.sample_margins[support])),
                affine_weights=fit_w.detach().cpu().numpy().astype(np.float32),
                affine_bias=float(fit_b.item()),
                centroid_low=np.mean(benchmark.train.X_low[support], axis=0).astype(np.float32),
                centroid_high=np.mean(benchmark.train.X_high[support], axis=0).astype(np.float32),
                mean_curvature=float(np.mean(atlas.sample_curvature[support])),
                mean_boundary_density=float(np.mean(atlas.sample_boundary_density[support])),
                mean_hardness=float(np.mean(atlas.sample_hardness[support])),
                mean_mergeability=float(np.mean([s for m in members for s in region_mergeability.get(m, [1.0])])),
            )
        )

    macro_adjacency_sets: dict[int, set[int]] = defaultdict(set)
    for region_id, neighbors in atlas.adjacency.items():
        macro_left = macro_id_map[uf.find(region_id)]
        for neigh in neighbors:
            macro_right = macro_id_map[uf.find(neigh)]
            if macro_left == macro_right:
                continue
            macro_adjacency_sets[macro_left].add(macro_right)
            macro_adjacency_sets[macro_right].add(macro_left)
    macro_adjacency = {mid: tuple(sorted(neigh)) for mid, neigh in macro_adjacency_sets.items()}
    for mid in range(len(macro_regions)):
        macro_adjacency.setdefault(mid, tuple())

    return MacroRegionAtlas(
        sample_macro_ids=sample_macro_ids,
        macro_regions=tuple(macro_regions),
        adjacency=macro_adjacency,
        merge_edge_count=merge_edge_count,
    )


def derive_target_architecture(
    cfg: HighDReluCompressionConfig,
    atlas: RegionAtlas,
    macro_atlas: MacroRegionAtlas,
    *,
    input_dim: int,
) -> CompressionArchitecture:
    source_params = _mlp_param_count(input_dim, cfg.source_hidden, cfg.source_depth)
    mean_degree = 0.0
    if macro_atlas.adjacency:
        mean_degree = float(np.mean([len(v) for v in macro_atlas.adjacency.values()]))
    num_macro = max(len(macro_atlas.macro_regions), 2)
    stable_fraction = np.mean(
        [
            (layer["always_on"] + layer["always_off"]) / max(layer["total"], 1)
            for layer in atlas.stable_neuron_stats
        ]
    )
    hidden_size = int(round(2.0 * math.sqrt(num_macro) + mean_degree + 4.0 * (1.0 - stable_fraction)))
    hidden_size = min(cfg.architecture_max_hidden, max(cfg.architecture_min_hidden, hidden_size))

    if num_macro <= 8:
        depth = min(2, cfg.source_depth)
    elif num_macro <= 24:
        depth = min(3, cfg.source_depth)
    else:
        depth = cfg.source_depth

    params = _mlp_param_count(input_dim, hidden_size, depth)
    while params >= source_params and hidden_size > cfg.architecture_min_hidden:
        hidden_size -= 1
        params = _mlp_param_count(input_dim, hidden_size, depth)

    return CompressionArchitecture(
        input_dim=input_dim,
        hidden_size=hidden_size,
        depth=depth,
        params=params,
    )


def _cluster_macro_regions(
    macro_atlas: MacroRegionAtlas,
    hidden_size: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    macro_regions = list(macro_atlas.macro_regions)
    feature_rows = []
    weights = []
    for region in macro_regions:
        jac = region.affine_weights
        jac_norm = np.linalg.norm(jac)
        if jac_norm > 0:
            jac = jac / jac_norm
        centroid = region.centroid_high
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        feature_rows.append(np.concatenate([jac, 0.35 * centroid]))
        weights.append(region.support_size * (1.0 + region.mean_hardness))
    features = np.asarray(feature_rows, dtype=np.float32)
    sample_weights = np.asarray(weights, dtype=np.float32)

    if len(features) >= hidden_size:
        kmeans = KMeans(n_clusters=hidden_size, n_init=10, random_state=seed)
        labels = kmeans.fit_predict(features, sample_weight=sample_weights)
    else:
        labels = np.arange(len(features))
        hidden_size = len(features)

    centers_w = []
    centers_b = []
    for cluster in range(hidden_size):
        members = [macro_regions[i] for i in range(len(macro_regions)) if labels[i] == cluster]
        cluster_weight = np.asarray(
            [m.support_size * (1.0 + m.mean_hardness) for m in members],
            dtype=np.float32,
        )
        affines = np.stack([m.affine_weights for m in members], axis=0)
        centroids = np.stack([m.centroid_high for m in members], axis=0)
        w = np.average(affines, axis=0, weights=cluster_weight)
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-6:
            w = w / w_norm
        centroid = np.average(centroids, axis=0, weights=cluster_weight)
        b = -float(np.dot(w, centroid))
        centers_w.append(w.astype(np.float32))
        centers_b.append(b)
    return np.stack(centers_w, axis=0), np.asarray(centers_b, dtype=np.float32)


def initialize_geometry_guided_mlp(
    model: MLP,
    macro_atlas: MacroRegionAtlas,
    *,
    seed: int,
) -> None:
    _set_seed(seed)
    linears = _linear_layers(model)
    hidden_size = linears[0].out_features
    prototype_w, prototype_b = _cluster_macro_regions(macro_atlas, hidden_size, seed=seed)

    with torch.no_grad():
        linears[0].weight.zero_()
        linears[0].bias.zero_()
        linears[0].weight[: len(prototype_w)] = torch.tensor(prototype_w, dtype=torch.float32)
        linears[0].bias[: len(prototype_b)] = torch.tensor(prototype_b, dtype=torch.float32)
        if len(prototype_w) < hidden_size:
            nn.init.kaiming_uniform_(linears[0].weight[len(prototype_w):], a=math.sqrt(5))

        for linear in linears[1:-1]:
            linear.weight.zero_()
            linear.bias.zero_()
            eye = torch.eye(linear.out_features, linear.in_features)
            linear.weight.copy_(eye)

        for linear in linears[-1:]:
            linear.weight.zero_()
            linear.bias.zero_()


def solve_output_layer_from_teacher(
    model: MLP, X_high: torch.Tensor, teacher_logits: torch.Tensor
) -> None:
    linears = _linear_layers(model)
    with torch.no_grad():
        _, _, _, hidden = _forward_hidden(model, X_high)
        fit_w, fit_b = _fit_affine_map(hidden, teacher_logits.squeeze(1))
        linears[-1].weight.copy_(fit_w.unsqueeze(0))
        linears[-1].bias.copy_(fit_b.unsqueeze(0))


def _importance_scores_from_source(model: MLP, X_high: torch.Tensor) -> list[torch.Tensor]:
    _, masks, hiddens, _ = _forward_hidden(model, X_high)
    linears = _linear_layers(model)
    scores = []
    for idx, hidden in enumerate(hiddens):
        act = hidden.abs().mean(dim=0)
        out_norm = linears[idx + 1].weight.detach().abs().mean(dim=0)
        mixed = (~masks[idx].all(dim=0) & masks[idx].any(dim=0)).float()
        stable_penalty = torch.where(masks[idx].all(dim=0), 0.5, 1.0)
        scores.append(act * out_norm * mixed * stable_penalty)
    return scores


def build_pruned_baseline(
    source_model: MLP,
    target_params: int,
    X_high: torch.Tensor,
) -> MLP:
    linears = _linear_layers(source_model)
    input_dim = linears[0].in_features
    source_depth = len(linears) - 1
    best_hidden = 2
    best_gap = float("inf")
    for hidden in range(2, linears[0].out_features + 1):
        params = _mlp_param_count(input_dim, hidden, source_depth)
        gap = abs(params - target_params)
        if gap < best_gap:
            best_hidden = hidden
            best_gap = gap
    pruned = _make_classifier(input_dim, best_hidden, source_depth)
    source_scores = _importance_scores_from_source(source_model, X_high)
    with torch.no_grad():
        src_linears = _linear_layers(source_model)
        dst_linears = _linear_layers(pruned)
        prev_idx = None
        for layer_idx in range(source_depth):
            scores = source_scores[layer_idx]
            keep = torch.topk(scores, k=best_hidden).indices.sort().values
            if layer_idx == 0:
                dst_linears[layer_idx].weight.copy_(src_linears[layer_idx].weight[keep])
            else:
                dst_linears[layer_idx].weight.copy_(src_linears[layer_idx].weight[keep][:, prev_idx])
            dst_linears[layer_idx].bias.copy_(src_linears[layer_idx].bias[keep])
            prev_idx = keep
        dst_linears[-1].weight.copy_(src_linears[-1].weight[:, prev_idx])
        dst_linears[-1].bias.copy_(src_linears[-1].bias)
    return pruned


def _sample_weights_from_atlas(atlas: RegionAtlas, device: torch.device) -> torch.Tensor:
    curvature = torch.tensor(_normalize_array(atlas.sample_curvature), dtype=torch.float32, device=device)
    boundary = torch.tensor(_normalize_array(atlas.sample_boundary_density), dtype=torch.float32, device=device)
    churn = torch.tensor(_normalize_array(atlas.sample_activation_churn), dtype=torch.float32, device=device)
    return 1.0 + 0.8 * curvature + 0.6 * boundary + 0.4 * churn


def _macro_representatives_low(
    macro_atlas: MacroRegionAtlas,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        np.stack([macro.centroid_low for macro in macro_atlas.macro_regions], axis=0),
        dtype=torch.float32,
        device=device,
    )


def _collect_metrics(
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


def _serialize(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items()}
    return obj


def run_single_high_d_compression(
    cfg: HighDReluCompressionConfig,
    *,
    observed_dim: int,
) -> dict[str, Any]:
    _set_seed(cfg.seed)
    device = _device_for(cfg)
    benchmark = make_embedded_classification_benchmark(
        dataset=cfg.dataset,
        observed_dim=observed_dim,
        n_samples=cfg.n_samples,
        noise=cfg.noise,
        n_turns=cfg.n_turns,
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
    t0 = time.time()
    source_trace = _train_classifier(
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
    source_train_time = time.time() - t0
    source_metrics = _collect_metrics(source_model, benchmark, device=device)
    source_metrics["train_time_s"] = source_train_time

    atlas = build_region_atlas(
        source_model,
        benchmark,
        device=device,
        knn_k=cfg.knn_k,
        curvature_eps=cfg.curvature_eps,
        boundary_eps=cfg.boundary_eps,
        boundary_perturbations=cfg.boundary_perturbations,
    )
    macro_atlas = coarsen_region_atlas(
        atlas,
        benchmark,
        max_affine_gap=cfg.max_affine_gap,
        max_merge_bce_increase=cfg.max_merge_bce_increase,
        min_merge_margin=cfg.min_merge_margin,
        low_complexity_quantile=cfg.low_complexity_quantile,
    )
    target_arch = derive_target_architecture(cfg, atlas, macro_atlas, input_dim=benchmark.observed_dim)

    source_logits_train = source_model(X_train).detach()
    sample_weights = _sample_weights_from_atlas(atlas, device)
    jacobian_points_low = _macro_representatives_low(macro_atlas, device)
    teacher_low_view = EmbeddedLowDimView(source_model, benchmark.embedding).to(device)
    teacher_jacobians_low = gradient(teacher_low_view, jacobian_points_low, chunk_size=256).detach()

    geometry_model = _make_classifier(
        benchmark.observed_dim, target_arch.hidden_size, target_arch.depth
    ).to(device)
    initialize_geometry_guided_mlp(geometry_model, macro_atlas, seed=cfg.seed)
    solve_output_layer_from_teacher(geometry_model, X_train, source_logits_train)
    geometry_low_view = EmbeddedLowDimView(geometry_model, benchmark.embedding).to(device)
    t0 = time.time()
    geometry_trace = _train_classifier(
        geometry_model,
        X_train,
        y_train,
        epochs=cfg.cleanup_epochs,
        lr=cfg.cleanup_lr,
        teacher_logits=source_logits_train,
        sample_weights=sample_weights,
        jacobian_points_low=jacobian_points_low,
        teacher_jacobians_low=teacher_jacobians_low,
        low_dim_view=geometry_low_view,
        distill_weight=cfg.geometry_distill_weight,
        jacobian_weight=cfg.geometry_jacobian_weight,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        trace_every_n_steps=cfg.trace_every_n_steps,
    )
    geometry_time = time.time() - t0

    distilled_model = _make_classifier(
        benchmark.observed_dim, target_arch.hidden_size, target_arch.depth
    ).to(device)
    distilled_trace = _train_classifier(
        distilled_model,
        X_train,
        y_train,
        epochs=cfg.baseline_epochs,
        lr=cfg.baseline_lr,
        teacher_logits=source_logits_train,
        distill_weight=cfg.distill_weight,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        trace_every_n_steps=cfg.trace_every_n_steps,
    )

    small_model = _make_classifier(
        benchmark.observed_dim, target_arch.hidden_size, target_arch.depth
    ).to(device)
    small_trace = _train_classifier(
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

    pruned_model = build_pruned_baseline(source_model, target_arch.params, X_train).to(device)
    pruned_trace = _train_classifier(
        pruned_model,
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

    metrics = {
        "source_mlp": source_metrics,
        "geometric_relu": _collect_metrics(geometry_model, benchmark, device=device),
        "distilled_relu": _collect_metrics(distilled_model, benchmark, device=device),
        "small_relu": _collect_metrics(small_model, benchmark, device=device),
        "pruned_source": _collect_metrics(pruned_model, benchmark, device=device),
    }
    metrics["geometric_relu"]["construction_time_s"] = geometry_time
    for vals in metrics.values():
        vals["compression_ratio"] = float(metrics["source_mlp"]["params"] / vals["params"])

    return {
        "config": _serialize(cfg),
        "benchmark": {
            "name": benchmark.name,
            "observed_dim": benchmark.observed_dim,
            "intrinsic_dim": benchmark.intrinsic_dim,
            "train_size": len(benchmark.train.y),
            "val_size": len(benchmark.val.y),
            "test_size": len(benchmark.test.y),
            "native_reference_only": bool(
                benchmark.observed_dim == benchmark.intrinsic_dim and benchmark.observed_dim == 2
            ),
        },
        "target_architecture": _serialize(target_arch),
        "atlas": {
            "n_regions": len(atlas.regions),
            "n_macro_regions": len(macro_atlas.macro_regions),
            "merge_edge_count": macro_atlas.merge_edge_count,
            "correlations": atlas.correlations,
            "stable_neuron_stats": _serialize(atlas.stable_neuron_stats),
        },
        "metrics": metrics,
        "training_traces": {
            "source_mlp": source_trace,
            "geometric_relu": geometry_trace,
            "distilled_relu": distilled_trace,
            "small_relu": small_trace,
            "pruned_source": pruned_trace,
        },
        "region_atlas": _serialize(atlas),
        "macro_region_atlas": _serialize(macro_atlas),
    }


def run_high_d_first_program(cfg: HighDReluCompressionConfig) -> dict[str, Any]:
    results = {}
    for observed_dim in cfg.observed_dims:
        results[str(observed_dim)] = run_single_high_d_compression(cfg, observed_dim=observed_dim)
    return {
        "program": "high_d_first_relu_geometric_compression",
        "results": results,
    }


def save_high_d_program_results(results: dict[str, Any], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
