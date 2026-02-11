from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import Axes, DashedVMobject, NumberPlane, VMobject


@dataclass(frozen=True)
class StoryConfig:
    bits: int = 4
    extent: float = 3.0
    grid_resolution: int = 81
    circle_radius: float = 1.5


CFG = StoryConfig()


def quantize_matrix(W: np.ndarray, delta: float) -> np.ndarray:
    return np.round(W / delta) * delta


# Narrative arc configuration: 3-layer 2->2 network with visible quantization error.
NARR_DELTA = 0.5  # 2-bit-like step for visual separation.
NARR_WS = [
    np.array([[0.8, 0.3], [-0.2, 0.9]], dtype=np.float64),
    np.array([[0.7, -0.4], [0.5, 0.6]], dtype=np.float64),
    np.array([[0.9, -0.1], [0.4, 0.8]], dtype=np.float64),
]
# NARR_BS = [
#     np.array([0.8, -0.1], dtype=np.float64),
#     np.array([-0.1, 0.1], dtype=np.float64),
#     np.array([-0.2, 0.1], dtype=np.float64),
# ]

NARR_BS = [
    np.array([0.0, 0.0], dtype=np.float64),
    np.array([0.0, 0.0], dtype=np.float64),
    np.array([0.0, 0.0], dtype=np.float64),
]
NARR_WS_Q = [quantize_matrix(W, NARR_DELTA) for W in NARR_WS]
# NARR_BS_Q = [quantize_matrix(b, NARR_DELTA) for b in NARR_BS]

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def as_column_vector(v: np.ndarray) -> np.ndarray:
    """Return v as a 2D column vector for matrix display utilities."""
    return np.asarray(v, dtype=np.float64).reshape(-1, 1)


def sample_circle(radius: float, n: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def polyline_from_points(
    axes: Axes | NumberPlane,
    points: np.ndarray,
    color,
    stroke_width: float = 2.0,
    dashed: bool = False,
    close_path: bool = False,
    opacity: float = 1.0,
) -> VMobject:
    pts = points
    if close_path:
        pts = np.vstack([points, points[0]])

    mobj = VMobject(stroke_color=color, stroke_width=stroke_width, stroke_opacity=opacity)
    mobj.set_points_as_corners([axes.c2p(float(x), float(y)) for x, y in pts])
    if dashed:
        return DashedVMobject(mobj, num_dashes=36)
    return mobj


def narr_fitted_axes(
    points_list: list[np.ndarray],
    size: float = 6.0,
    pad_frac: float = 0.15,
) -> Axes:
    all_pts = np.vstack(points_list)
    xmin, xmax = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
    ymin, ymax = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
    span = max(xmax - xmin, ymax - ymin, 0.5)
    pad = pad_frac * span
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = span / 2 + pad
    return Axes(
        x_range=[cx - half, cx + half, half],
        y_range=[cy - half, cy + half, half],
        axis_config={"stroke_width": 1, "stroke_opacity": 0.25},
        x_length=size,
        y_length=size,
        tips=False,
    )


def narr_global_axes(
    points_list: list[np.ndarray],
    size: float = 6.5,
    pad_frac: float = 0.2,
) -> Axes:
    all_pts = np.vstack(points_list)
    xmin, xmax = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
    ymin, ymax = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())

    xmin = min(xmin, 0.0)
    xmax = max(xmax, 0.0)
    ymin = min(ymin, 0.0)
    ymax = max(ymax, 0.0)

    span = max(xmax - xmin, ymax - ymin, 0.5)
    pad = pad_frac * span
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = span / 2 + pad

    return Axes(
        x_range=[cx - half, cx + half, half],
        y_range=[cy - half, cy + half, half],
        axis_config={"stroke_width": 1.2, "stroke_opacity": 0.35},
        x_length=size,
        y_length=size,
        tips=False,
    )


def narr_sign_pattern(points: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = points @ W.T + b
    bits = (z > 0.0).astype(np.int32)
    return bits[:, 0] * 2 + bits[:, 1]


def build_float_progression_steps(
    points: np.ndarray,
    ws: list[np.ndarray] | None = None,
    bs: list[np.ndarray] | None = None,
) -> list[tuple[str, np.ndarray, str]]:
    if ws is None:
        ws = NARR_WS
    if bs is None:
        bs = NARR_BS

    steps: list[tuple[str, np.ndarray, str]] = [("input manifold a0", points.copy(), "input")]
    a = points.copy()
    for idx, (W, b) in enumerate(zip(ws, bs), start=1):
        z = a @ W.T + b
        steps.append((f"apply W{idx}, b{idx}:  z{idx} = W{idx} a{idx-1} + b{idx}", z.copy(), "affine"))
        a = relu(z)
        steps.append((f"apply ReLU:  a{idx} = max(z{idx}, 0)", a.copy(), "relu"))
    return steps


def build_float_quant_progression(points: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str]]:
    f_manifolds = [points.copy()]
    q_manifolds = [points.copy()]
    labels = ["input (shared)"]
    kinds = ["input"]

    a_f, a_q = points.copy(), points.copy()
    for idx, (W, Wq, b) in enumerate(zip(NARR_WS, NARR_WS_Q, NARR_BS), start=1):
        zf = a_f @ W.T + b
        zq = a_q @ Wq.T + b
        f_manifolds.append(zf.copy())
        q_manifolds.append(zq.copy())
        labels.append(f"after W{idx}, b{idx}")
        kinds.append("affine")

        a_f = relu(zf)
        a_q = relu(zq)
        f_manifolds.append(a_f.copy())
        q_manifolds.append(a_q.copy())
        labels.append(f"after ReLU {idx}")
        kinds.append("relu")

    return f_manifolds, q_manifolds, labels, kinds


def fit_affine_transport(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    augmented = np.column_stack([source, np.ones(source.shape[0])])
    sol, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    linear = sol[:2, :]
    offset = sol[2, :]
    return linear, offset


def apply_affine_transport(points: np.ndarray, linear: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return points @ linear + offset


def forward_three_layer_float_quant(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_f, a_q = points.copy(), points.copy()
    topo_mask = np.zeros(len(points), dtype=bool)

    for W, Wq, b in zip(NARR_WS, NARR_WS_Q, NARR_BS):
        zf = a_f @ W.T + b
        zq = a_q @ Wq.T + b

        pat_f = narr_sign_pattern(a_f, W, b)
        pat_q = narr_sign_pattern(a_q, Wq, b)
        topo_mask |= (pat_f != pat_q)

        a_f = relu(zf)
        a_q = relu(zq)

    y_f, y_q = a_f, a_q
    metric_mask = ~topo_mask
    return y_f, y_q, metric_mask, topo_mask
