"""Shared primitives for quantization geometry notebooks."""

from aleph.qgeom.core import (
    AllExperimentStats,
    ExperimentStats,
    LayerGeometryReport,
    LayerStats,
    RunGeometryReport,
    quantize,
)
from aleph.qgeom.drawing_2d import draw_polygon, set_fixed_scale
from aleph.qgeom.drawing_3d import draw_box_3d, draw_vertices_and_hull_3d, draw_wireframe_box
from aleph.qgeom.experiment import run_all_manifolds, run_experiment
from aleph.qgeom.geometry_2d import (
    compute_polygon_area,
    get_box_vertices_2d,
    minkowski_sum_2d,
    transform_vertices,
)
from aleph.qgeom.geometry_3d import get_hypercube_vertices, minkowski_sum_3d
from aleph.qgeom.manifolds import compute_manifold_errors, compute_pointwise_errors, make_manifold
from aleph.qgeom.metrics import (
    build_layer_geometry_report,
    compute_correctability_score,
    compute_entropy_metrics,
    compute_fate_metrics,
    compute_geometry_metrics,
)
from aleph.qgeom.policy import (
    PolicySimulationResult,
    build_baseline_policy,
    evenly_spaced_points,
    score_correction_points,
    simulate_policy,
)
from aleph.qgeom.canonical import (
    CanonicalSpaceTracker,
    ForwardTrace,
    ReLUDisagreementTracker,
    error_attribution,
)
from aleph.qgeom.transformer_analysis import (
    collect_transformer_traces,
    collect_transformer_layer_reports,
    quantize_decompose_tensor,
    run_report_to_layer_gain_map,
)

__all__ = [
    "AllExperimentStats",
    "ExperimentStats",
    "LayerGeometryReport",
    "LayerStats",
    "RunGeometryReport",
    "quantize",
    "get_box_vertices_2d",
    "minkowski_sum_2d",
    "compute_polygon_area",
    "transform_vertices",
    "get_hypercube_vertices",
    "minkowski_sum_3d",
    "draw_polygon",
    "set_fixed_scale",
    "draw_box_3d",
    "draw_vertices_and_hull_3d",
    "draw_wireframe_box",
    "make_manifold",
    "compute_pointwise_errors",
    "compute_manifold_errors",
    "run_experiment",
    "run_all_manifolds",
    "compute_geometry_metrics",
    "compute_entropy_metrics",
    "compute_fate_metrics",
    "compute_correctability_score",
    "build_layer_geometry_report",
    "score_correction_points",
    "build_baseline_policy",
    "evenly_spaced_points",
    "simulate_policy",
    "PolicySimulationResult",
    "collect_transformer_traces",
    "collect_transformer_layer_reports",
    "quantize_decompose_tensor",
    "run_report_to_layer_gain_map",
    "CanonicalSpaceTracker",
    "ForwardTrace",
    "ReLUDisagreementTracker",
    "error_attribution",
]
