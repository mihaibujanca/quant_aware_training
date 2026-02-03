"""Constants, dataclasses, and base quantization for geometry experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


# ============================================================
# Legacy data structures (used by existing geometry notebooks)
# ============================================================


@dataclass
class LayerStats:
    """Statistics for a single layer."""

    layer_idx: int
    weight_matrix: np.ndarray
    spectral_norm: float
    determinant: float
    condition_number: float
    error_half_widths: np.ndarray
    error_volume: float


@dataclass
class ExperimentStats:
    """Statistics for a full experiment."""

    name: str
    input_point: np.ndarray
    layer_stats: List[LayerStats]
    cumulative_error_vertices: np.ndarray
    cumulative_error_volume: float
    bounding_box: np.ndarray
    relative_error: np.ndarray

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "input": self.input_point.tolist(),
            "final_volume": self.cumulative_error_volume,
            "bbox": self.bounding_box.tolist(),
            "relative_error": self.relative_error.tolist(),
            "spectral_norms": [ls.spectral_norm for ls in self.layer_stats],
        }


@dataclass
class AllExperimentStats:
    """Container for all experiment results."""

    experiments: Dict[str, ExperimentStats] = field(default_factory=dict)

    def add(self, stats: ExperimentStats) -> None:
        self.experiments[stats.name] = stats

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("SUMMARY OF ALL EXPERIMENTS")
        print("=" * 70)
        for name, stats in self.experiments.items():
            print(f"\n{name}:")
            print(f"  Input: {stats.input_point}")
            print(f"  Final error volume: {stats.cumulative_error_volume:.6f}")
            print(f"  Bounding box: {stats.bounding_box}")
            print(f"  Relative error: {stats.relative_error}")


# ============================================================
# Geometry-guided correction report schema
# ============================================================


@dataclass
class LayerGeometryReport:
    """Per-layer report for geometry-guided quantization diagnostics."""

    layer_index: int
    linear_error_norm: float
    nonlinear_error_norm: float
    saturation_count: int
    relu_flip_rate: float
    survive_rate: float
    dead_rate: float
    anisotropy_ratio: float
    entropy_proxy: float
    volume_proxy: float
    correctability_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_error_norm(self) -> float:
        return float(self.linear_error_norm + self.nonlinear_error_norm)

    @property
    def linear_fraction(self) -> float:
        denom = self.total_error_norm
        if denom <= 1e-12:
            return 1.0
        return float(self.linear_error_norm / denom)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "linear_error_norm": self.linear_error_norm,
            "nonlinear_error_norm": self.nonlinear_error_norm,
            "saturation_count": self.saturation_count,
            "relu_flip_rate": self.relu_flip_rate,
            "survive_rate": self.survive_rate,
            "dead_rate": self.dead_rate,
            "anisotropy_ratio": self.anisotropy_ratio,
            "entropy_proxy": self.entropy_proxy,
            "volume_proxy": self.volume_proxy,
            "correctability_score": self.correctability_score,
            "linear_fraction": self.linear_fraction,
            "metadata": self.metadata,
        }


@dataclass
class RunGeometryReport:
    """Run-level report that aggregates per-layer geometry diagnostics."""

    model_name: str
    task_name: str
    bit_width: int
    layer_reports: List[LayerGeometryReport]
    policy_recommendations: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def aggregate_metrics(self) -> Dict[str, float]:
        if not self.layer_reports:
            return {
                "mean_linear_error_norm": 0.0,
                "mean_nonlinear_error_norm": 0.0,
                "mean_linear_fraction": 1.0,
                "mean_correctability_score": 0.0,
                "mean_relu_flip_rate": 0.0,
                "mean_survive_rate": 0.0,
                "mean_dead_rate": 0.0,
                "total_saturation_count": 0.0,
            }

        linear = np.array([r.linear_error_norm for r in self.layer_reports], dtype=float)
        nonlinear = np.array([r.nonlinear_error_norm for r in self.layer_reports], dtype=float)
        flips = np.array([r.relu_flip_rate for r in self.layer_reports], dtype=float)
        survive = np.array([r.survive_rate for r in self.layer_reports], dtype=float)
        dead = np.array([r.dead_rate for r in self.layer_reports], dtype=float)
        correctability = np.array([r.correctability_score for r in self.layer_reports], dtype=float)
        linear_fraction = np.array([r.linear_fraction for r in self.layer_reports], dtype=float)
        saturation = np.array([r.saturation_count for r in self.layer_reports], dtype=float)

        return {
            "mean_linear_error_norm": float(linear.mean()),
            "mean_nonlinear_error_norm": float(nonlinear.mean()),
            "mean_linear_fraction": float(linear_fraction.mean()),
            "mean_correctability_score": float(correctability.mean()),
            "mean_relu_flip_rate": float(flips.mean()),
            "mean_survive_rate": float(survive.mean()),
            "mean_dead_rate": float(dead.mean()),
            "total_saturation_count": float(saturation.sum()),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_name": self.task_name,
            "bit_width": self.bit_width,
            "policy_recommendations": self.policy_recommendations,
            "aggregate_metrics": self.aggregate_metrics(),
            "metadata": self.metadata,
            "layer_reports": [r.to_dict() for r in self.layer_reports],
        }


# ============================================================
# Quantization
# ============================================================


def quantize(W: np.ndarray, delta: float) -> np.ndarray:
    """Quantize matrix to nearest grid point."""

    return np.round(W / delta) * delta
