"""Generic experiment runners for quantization geometry."""

import numpy as np
from aleph.qgeom.core import LayerStats, ExperimentStats, AllExperimentStats, quantize
from aleph.qgeom.geometry_2d import transform_vertices, minkowski_sum_2d, compute_polygon_area
from aleph.qgeom.manifolds import make_manifold, compute_manifold_errors


def run_experiment(name, x_input, weight_matrices, compute_error_fn, delta, all_stats=None):
    """Run a 2D experiment and collect statistics.

    Args:
        name: Experiment name
        x_input: Input point (2D)
        weight_matrices: List of weight matrices (true, pre-quantization)
        compute_error_fn: Function to compute error vertices at each layer
        delta: Quantization step size
        all_stats: Optional AllExperimentStats to register results in

    Returns:
        ExperimentStats object, quantized weights
    """
    quant_weights = [quantize(W, delta) for W in weight_matrices]

    layer_stats = []
    val = x_input.copy()
    cumulative_W = np.eye(2)
    cumulative_error_vertices = None

    for i, (W_true, W) in enumerate(zip(weight_matrices, quant_weights)):
        spectral_norm = np.linalg.norm(W, ord=2)
        det = np.linalg.det(W)
        svd = np.linalg.svd(W, compute_uv=False)
        cond = svd.max() / svd.min() if svd.min() > 0 else np.inf

        local_error_vertices = compute_error_fn(val, W, delta)

        cumulative_W_after = W @ cumulative_W
        try:
            inv_W = np.linalg.inv(cumulative_W_after)
            error_vertices_input = transform_vertices(local_error_vertices, inv_W)
        except:
            error_vertices_input = local_error_vertices

        if cumulative_error_vertices is None:
            cumulative_error_vertices = error_vertices_input
        else:
            cumulative_error_vertices = minkowski_sum_2d(
                cumulative_error_vertices, error_vertices_input
            )

        hw = np.abs(error_vertices_input).max(axis=0)

        layer_stats.append(LayerStats(
            layer_idx=i,
            weight_matrix=W.copy(),
            spectral_norm=spectral_norm,
            determinant=det,
            condition_number=cond,
            error_half_widths=hw,
            error_volume=compute_polygon_area(error_vertices_input)
        ))

        val = W @ val
        cumulative_W = cumulative_W_after

    bbox_min = cumulative_error_vertices.min(axis=0)
    bbox_max = cumulative_error_vertices.max(axis=0)
    rel_error = (bbox_max - bbox_min) / (2 * np.abs(x_input) + 1e-10)

    stats = ExperimentStats(
        name=name,
        input_point=x_input.copy(),
        layer_stats=layer_stats,
        cumulative_error_vertices=cumulative_error_vertices,
        cumulative_error_volume=compute_polygon_area(cumulative_error_vertices),
        bounding_box=np.array([bbox_min, bbox_max]),
        relative_error=rel_error
    )

    if all_stats is not None:
        all_stats.add(stats)
    return stats, quant_weights


def run_all_manifolds(quant_weights, manifold_names=None, n_points=48, *, delta):
    """Run error analysis across multiple manifolds.

    Args:
        quant_weights: List of quantized weight matrices
        manifold_names: List of manifold names to test
        n_points: Number of points per manifold
        delta: Quantization step size (keyword-only)

    Returns dict mapping manifold name to {points, metadata, results, stats}.
    """
    if manifold_names is None:
        manifold_names = ['circle', 'ellipse', 'line', 'spiral', 'figure_eight', 'two_blobs']

    all_results = {}
    for name in manifold_names:
        print(f"  Processing manifold: {name}")
        points, metadata = make_manifold(name, n_points=n_points)
        results = compute_manifold_errors(points, quant_weights, delta)

        magnitudes = [r['error_magnitude'] for r in results]
        l1_norms = [r['l1_norm'] for r in results]

        all_results[name] = {
            'points': points,
            'metadata': metadata,
            'results': results,
            'stats': {
                'error_mag_min': np.min(magnitudes),
                'error_mag_max': np.max(magnitudes),
                'error_mag_mean': np.mean(magnitudes),
                'error_mag_std': np.std(magnitudes),
                'variation_ratio': np.max(magnitudes) / np.min(magnitudes),
                'correlation_l1': np.corrcoef(l1_norms, magnitudes)[0, 1]
            }
        }
    return all_results
