"""3D geometry primitives: hypercube vertices, Minkowski sums."""

import numpy as np
from itertools import product
from scipy.spatial import ConvexHull


def get_hypercube_vertices(half_width, dims=3):
    """Get vertices of a hypercube centered at origin."""
    return np.array(list(product([-1, 1], repeat=dims))) * half_width


def minkowski_sum_3d(V1, V2):
    """Minkowski sum of two 3D vertex sets."""
    sums = np.array([v1 + v2 for v1 in V1 for v2 in V2])
    if len(sums) > 4:
        try:
            hull = ConvexHull(sums)
            return sums[hull.vertices]
        except:
            return sums
    return sums
