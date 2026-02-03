"""3D matplotlib drawing utilities for quantization geometry plots."""

import numpy as np
from itertools import product
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_box_3d(ax, center, half_widths, color, alpha=0.3, label=None):
    """Draw a 3D box centered at 'center' with given half-widths."""
    hw = np.array(half_widths)
    vertices = np.array(list(product([-1, 1], repeat=3))) * hw + center
    faces = [
        [vertices[0], vertices[1], vertices[3], vertices[2]],
        [vertices[4], vertices[5], vertices[7], vertices[6]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[2], vertices[6], vertices[4]],
        [vertices[1], vertices[3], vertices[7], vertices[5]],
    ]
    ax.add_collection3d(Poly3DCollection(
        faces, alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.5
    ))
    if label:
        ax.text(center[0], center[1], center[2] + hw[2] * 1.2, label, fontsize=10)


def draw_vertices_and_hull_3d(ax, vertices, color, alpha=0.3):
    """Draw vertices and their convex hull in 3D."""
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               c=color, s=20, alpha=0.8)
    if len(vertices) >= 4:
        try:
            hull = ConvexHull(vertices)
            for simplex in hull.simplices:
                triangle = vertices[simplex]
                ax.add_collection3d(Poly3DCollection(
                    [triangle], alpha=alpha, facecolor=color,
                    edgecolor='black', linewidth=0.5
                ))
        except:
            pass


def draw_wireframe_box(ax, half_widths, color='gray', alpha=0.5):
    """Draw wireframe of axis-aligned box for reference."""
    hw = half_widths
    edges = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            edges.append([[-hw[0]*i, -hw[1]*j, -hw[2]], [-hw[0]*i, -hw[1]*j, hw[2]]])
            edges.append([[-hw[0]*i, -hw[1], -hw[2]*j], [-hw[0]*i, hw[1], -hw[2]*j]])
            edges.append([[-hw[0], -hw[1]*i, -hw[2]*j], [hw[0], -hw[1]*i, -hw[2]*j]])
    for edge in edges:
        ax.plot3D(*zip(*edge), color=color, alpha=alpha, linewidth=1)
