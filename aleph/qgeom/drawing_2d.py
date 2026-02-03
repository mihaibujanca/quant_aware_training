"""2D matplotlib drawing utilities for quantization geometry plots."""

import numpy as np
from scipy.spatial import ConvexHull


def draw_polygon(ax, vertices, color, alpha=0.3, edgecolor=None, linewidth=2, label=None):
    """Draw a polygon from vertices."""
    if len(vertices) < 3:
        ax.scatter(vertices[:, 0], vertices[:, 1], c=color, s=50, label=label)
        return
    try:
        hull = ConvexHull(vertices)
        hull_verts = vertices[hull.vertices]
        hull_verts = np.vstack([hull_verts, hull_verts[0]])
        ax.fill(hull_verts[:, 0], hull_verts[:, 1], color=color, alpha=alpha, label=label)
        ax.plot(hull_verts[:, 0], hull_verts[:, 1], color=edgecolor or color, linewidth=linewidth)
    except:
        ax.scatter(vertices[:, 0], vertices[:, 1], c=color, s=50, label=label)


def set_fixed_scale(ax, scale, center=(0, 0)):
    """Set fixed axis limits with equal aspect ratio."""
    ax.set_xlim(center[0] - scale, center[0] + scale)
    ax.set_ylim(center[1] - scale, center[1] + scale)
    ax.set_aspect('equal')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
