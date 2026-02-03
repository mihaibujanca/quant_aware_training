"""Geometry/entropy/fate metrics for quantization analysis."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from aleph.qgeom.core import LayerGeometryReport


def _as_2d(x: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr


def _covariance_with_shrinkage(x: np.ndarray, shrinkage: float = 1e-5) -> np.ndarray:
    x = _as_2d(x)
    n, d = x.shape
    if n <= 1:
        return np.eye(d, dtype=float) * shrinkage

    centered = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0
    cov += np.eye(cov.shape[0], dtype=float) * shrinkage
    return cov


def compute_geometry_metrics(
    linear_error: np.ndarray,
    nonlinear_error: np.ndarray,
    *,
    operator_matrix: np.ndarray | None = None,
    prev_total_error: np.ndarray | None = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute geometry-oriented metrics for one quantization point/layer."""

    linear = _as_2d(linear_error)
    nonlinear = _as_2d(nonlinear_error)
    total = linear + nonlinear

    linear_norm = float(np.linalg.norm(linear, axis=1).mean())
    nonlinear_norm = float(np.linalg.norm(nonlinear, axis=1).mean())
    total_norm = float(np.linalg.norm(total, axis=1).mean())

    cov = _covariance_with_shrinkage(total)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, eps, None)

    max_eig = float(np.max(eigvals))
    min_eig = float(np.min(eigvals))
    anisotropy_ratio = float(max_eig / max(min_eig, eps))

    # Volume proxy: sqrt(det(cov)) in log space for stability.
    logdet = float(np.sum(np.log(eigvals)))
    volume_proxy = float(np.exp(np.clip(0.5 * logdet, -80.0, 80.0)))

    if operator_matrix is not None:
        amplification = float(np.linalg.norm(operator_matrix, ord=2))
    elif prev_total_error is not None:
        prev = _as_2d(prev_total_error)
        prev_norm = float(np.linalg.norm(prev, axis=1).mean())
        amplification = float(total_norm / max(prev_norm, eps))
    else:
        amplification = 1.0

    return {
        "linear_error_norm": linear_norm,
        "nonlinear_error_norm": nonlinear_norm,
        "total_error_norm": total_norm,
        "anisotropy_ratio": anisotropy_ratio,
        "volume_proxy": volume_proxy,
        "amplification": amplification,
    }


def compute_entropy_metrics(
    error_samples: np.ndarray,
    *,
    shrinkage: float = 1e-5,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute entropy-style proxies with rank-aware robust fallbacks."""

    x = _as_2d(error_samples)
    n, d = x.shape

    cov = _covariance_with_shrinkage(x, shrinkage=shrinkage)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, eps, None)

    # Differential entropy proxy of Gaussian in bits.
    log_det = float(np.sum(np.log(eigvals)))
    entropy_proxy = float(0.5 * (d * math.log(2.0 * math.pi * math.e) + log_det) / math.log(2.0))

    # Effective rank of covariance spectrum.
    p = eigvals / np.sum(eigvals)
    spectral_entropy = -float(np.sum(p * np.log(np.clip(p, eps, None))))
    effective_rank = float(np.exp(spectral_entropy))

    volume_proxy = float(np.exp(np.clip(0.5 * log_det, -80.0, 80.0)))

    return {
        "entropy_proxy": entropy_proxy,
        "log_det_cov": log_det,
        "effective_rank": effective_rank,
        "volume_proxy": volume_proxy,
        "n_samples": float(n),
        "n_dims": float(d),
    }


def compute_fate_metrics(
    float_pre_activation: np.ndarray,
    quant_pre_activation: np.ndarray,
    *,
    saturation_mask: np.ndarray | None = None,
    threshold: float = 0.0,
) -> Dict[str, float]:
    """Compute survive/flip/dead rates and collapse mass."""

    fp = _as_2d(float_pre_activation)
    qp = _as_2d(quant_pre_activation)
    if fp.shape != qp.shape:
        raise ValueError(f"Shape mismatch: float={fp.shape}, quant={qp.shape}")

    fp_pos = fp > threshold
    qp_pos = qp > threshold

    survive = fp_pos & qp_pos
    dead = (~fp_pos) & (~qp_pos)
    flip = fp_pos ^ qp_pos

    n = float(fp.size)
    survive_rate = float(survive.sum() / n)
    dead_rate = float(dead.sum() / n)
    relu_flip_rate = float(flip.sum() / n)

    if saturation_mask is None:
        sat_count = 0
    else:
        sat_count = int(np.asarray(saturation_mask, dtype=bool).sum())

    collapse_mass = float(dead_rate + relu_flip_rate)

    return {
        "survive_rate": survive_rate,
        "dead_rate": dead_rate,
        "relu_flip_rate": relu_flip_rate,
        "saturation_count": sat_count,
        "collapse_mass": collapse_mass,
    }


def compute_correctability_score(
    *,
    linear_error_norm: float,
    nonlinear_error_norm: float,
    relu_flip_rate: float,
    saturation_count: int,
    n_elements: int,
    anisotropy_ratio: float,
    amplification: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """Compute [0, 1] score for expected linear correctability."""

    linear_fraction = linear_error_norm / max(linear_error_norm + nonlinear_error_norm, eps)
    sat_rate = saturation_count / max(n_elements, 1)

    anis_penalty = 1.0 / (1.0 + math.log1p(max(anisotropy_ratio - 1.0, 0.0)))
    amp_penalty = 1.0 / (1.0 + max(amplification - 1.0, 0.0))

    score = (
        0.55 * linear_fraction
        + 0.15 * (1.0 - relu_flip_rate)
        + 0.15 * (1.0 - sat_rate)
        + 0.10 * anis_penalty
        + 0.05 * amp_penalty
    )
    return float(np.clip(score, 0.0, 1.0))


def build_layer_geometry_report(
    layer_index: int,
    *,
    linear_error: np.ndarray,
    nonlinear_error: np.ndarray,
    float_pre_activation: np.ndarray,
    quant_pre_activation: np.ndarray,
    saturation_mask: np.ndarray | None = None,
    operator_matrix: np.ndarray | None = None,
    prev_total_error: np.ndarray | None = None,
    metadata: Dict[str, Any] | None = None,
) -> LayerGeometryReport:
    """Compose all metric families into a single layer report."""

    geometry = compute_geometry_metrics(
        linear_error,
        nonlinear_error,
        operator_matrix=operator_matrix,
        prev_total_error=prev_total_error,
    )
    entropy = compute_entropy_metrics(_as_2d(linear_error) + _as_2d(nonlinear_error))
    fate = compute_fate_metrics(
        float_pre_activation,
        quant_pre_activation,
        saturation_mask=saturation_mask,
    )

    n_elements = int(_as_2d(float_pre_activation).size)
    score = compute_correctability_score(
        linear_error_norm=geometry["linear_error_norm"],
        nonlinear_error_norm=geometry["nonlinear_error_norm"],
        relu_flip_rate=fate["relu_flip_rate"],
        saturation_count=int(fate["saturation_count"]),
        n_elements=n_elements,
        anisotropy_ratio=geometry["anisotropy_ratio"],
        amplification=geometry["amplification"],
    )

    extra: Dict[str, Any] = {
        "amplification": geometry["amplification"],
        "log_det_cov": entropy["log_det_cov"],
        "effective_rank": entropy["effective_rank"],
        "collapse_mass": fate["collapse_mass"],
        "n_samples": entropy["n_samples"],
        "n_dims": entropy["n_dims"],
    }
    if metadata:
        extra.update(metadata)

    return LayerGeometryReport(
        layer_index=layer_index,
        linear_error_norm=geometry["linear_error_norm"],
        nonlinear_error_norm=geometry["nonlinear_error_norm"],
        saturation_count=int(fate["saturation_count"]),
        relu_flip_rate=fate["relu_flip_rate"],
        survive_rate=fate["survive_rate"],
        dead_rate=fate["dead_rate"],
        anisotropy_ratio=geometry["anisotropy_ratio"],
        entropy_proxy=entropy["entropy_proxy"],
        volume_proxy=geometry["volume_proxy"],
        correctability_score=score,
        metadata=extra,
    )
