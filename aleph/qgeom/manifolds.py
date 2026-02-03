"""Manifold generation and error computation for quantization geometry."""

import numpy as np
from aleph.qgeom.geometry_2d import (
    get_box_vertices_2d, minkowski_sum_2d, compute_polygon_area, transform_vertices
)


def make_manifold(name, n_points=32, **kwargs):
    """Generate points on various 2D manifolds.

    Supported manifolds:
        circle, ellipse, line, spiral, figure_eight, two_blobs, grid

    Returns:
        (points, metadata) where points is (n_points, 2) and metadata
        includes 'type' ('closed', 'open', 'clusters', 'area').
    """
    if name == "circle":
        radius = kwargs.get('radius', 20)
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
        metadata = {'radius': radius, 'type': 'closed'}
    elif name == "ellipse":
        a, b = kwargs.get('a', 25), kwargs.get('b', 10)
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        points = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
        metadata = {'a': a, 'b': b, 'type': 'closed'}
    elif name == "line":
        start = np.array(kwargs.get('start', [-25, -10]))
        end = np.array(kwargs.get('end', [25, 10]))
        t = np.linspace(0, 1, n_points)
        points = start + t[:, np.newaxis] * (end - start)
        metadata = {'type': 'open'}
    elif name == "spiral":
        turns = kwargs.get('turns', 2)
        r_min, r_max = kwargs.get('r_min', 5), kwargs.get('r_max', 25)
        theta = np.linspace(0, turns * 2 * np.pi, n_points)
        r = np.linspace(r_min, r_max, n_points)
        points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        metadata = {'type': 'open'}
    elif name == "figure_eight":
        scale = kwargs.get('scale', 15)
        t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        points = np.column_stack([scale * np.sin(t), scale * np.sin(t) * np.cos(t)])
        metadata = {'type': 'closed'}
    elif name == "two_blobs":
        n_each = n_points // 2
        c1 = np.array(kwargs.get('center1', [-15, 0]))
        c2 = np.array(kwargs.get('center2', [15, 0]))
        std = kwargs.get('std', 5)
        blob1 = np.random.randn(n_each, 2) * std + c1
        blob2 = np.random.randn(n_points - n_each, 2) * std + c2
        points = np.vstack([blob1, blob2])
        metadata = {'type': 'clusters'}
    elif name == "grid":
        extent = kwargs.get('extent', 25)
        n_side = int(np.sqrt(n_points))
        x = np.linspace(-extent, extent, n_side)
        y = np.linspace(-extent, extent, n_side)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        metadata = {'extent': extent, 'n_side': n_side, 'type': 'area'}
    else:
        raise ValueError(f"Unknown manifold: {name}")
    return points, metadata


def compute_pointwise_errors(points, true_weights, quant_weights):
    """Compute the actual error vector for each input point.

    Since the network has no activations, the error is a linear function:
    error(x) = (Q_n...Q_1 - W_n...W_1) @ x

    Returns (errors, W_error) where errors is (n_points, 2) and
    W_error is the 2x2 error transform matrix.
    """
    W_float = np.eye(2)
    W_quant = np.eye(2)
    for Wt, Wq in zip(true_weights, quant_weights):
        W_float = Wt @ W_float
        W_quant = Wq @ W_quant
    W_error = W_quant - W_float
    errors = points @ W_error.T
    return errors, W_error


def compute_manifold_errors(points, quant_weights, delta):
    """Compute error region statistics for all points on a manifold.

    For each point, traces the error region (Minkowski sum of per-layer
    error boxes) through the network layers.
    """
    results = []
    for x in points:
        val = x.copy()
        cumulative_W = np.eye(2)
        cumulative_error_vertices = None

        for W in quant_weights:
            l1_norm = np.sum(np.abs(val))
            hw = (delta / 2) * l1_norm
            local_vertices = get_box_vertices_2d([hw, hw])

            cumulative_W_after = W @ cumulative_W
            try:
                inv_W = np.linalg.inv(cumulative_W_after)
                error_vertices_input = transform_vertices(local_vertices, inv_W)
            except:
                error_vertices_input = local_vertices

            if cumulative_error_vertices is None:
                cumulative_error_vertices = error_vertices_input
            else:
                cumulative_error_vertices = minkowski_sum_2d(
                    cumulative_error_vertices, error_vertices_input
                )
            val = W @ val
            cumulative_W = cumulative_W_after

        results.append({
            'input': x.copy(),
            'error_vertices': cumulative_error_vertices.copy(),
            'error_magnitude': np.max(np.linalg.norm(cumulative_error_vertices, axis=1)),
            'error_volume': compute_polygon_area(cumulative_error_vertices),
            'l1_norm': np.sum(np.abs(x)),
            'l2_norm': np.linalg.norm(x)
        })
    return results
