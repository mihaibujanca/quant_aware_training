"""2D geometry primitives: vertices, Minkowski sums, polygon area, transforms."""

import numpy as np
from scipy.spatial import ConvexHull


def get_box_vertices_2d(half_widths):
    """Get vertices of 2D box centered at origin."""
    hw = np.array(half_widths)
    return np.array([[-hw[0], -hw[1]], [-hw[0], hw[1]],
                     [hw[0], hw[1]], [hw[0], -hw[1]]])


def minkowski_sum_2d(V1, V2):
    """Minkowski sum of two 2D vertex sets."""
    sums = np.array([v1 + v2 for v1 in V1 for v2 in V2])
    if len(sums) >= 3:
        try:
            hull = ConvexHull(sums)
            return sums[hull.vertices]
        except:
            pass
    return sums


def compute_polygon_area(vertices):
    """Compute area of polygon using convex hull."""
    if len(vertices) < 3:
        return 0.0
    try:
        hull = ConvexHull(vertices)
        return hull.volume  # In 2D, 'volume' is area
    except:
        return 0.0


def transform_vertices(vertices, W):
    """Apply linear transformation W to vertices."""
    return vertices @ W.T
