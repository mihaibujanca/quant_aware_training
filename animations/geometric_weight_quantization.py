"""Manim scenes for geometric intuition of weight quantization.

This file mirrors the narrative in docs/geometric_error_correction_brief.md and
focuses on weight-only quantization for a small 2->2->2 ReLU network.

Render examples:
    manim -pqh animations/geometric_weight_quantization.py OpeningScene
    manim -pqh animations/geometric_weight_quantization.py PartitionShiftScene
    manim -pqh animations/geometric_weight_quantization.py GridDistortionScene
    manim -pqh animations/geometric_weight_quantization.py ErrorDecompositionScene
    manim -pqh animations/geometric_weight_quantization.py CorrectionCascadeScene
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import (
    Arrow,
    Axes,
    BLUE_D,
    Create,
    DashedVMobject,
    Dot,
    DOWN,
    FadeIn,
    FadeOut,
    GREEN_D,
    GREY_B,
    LEFT,
    NumberPlane,
    ORANGE,
    RED_D,
    RIGHT,
    Scene,
    Text,
    Transform,
    UP,
    VGroup,
    VMobject,
    WHITE,
    Write,
)


@dataclass(frozen=True)
class StoryConfig:
    bits: int = 4
    extent: float = 3.0
    grid_resolution: int = 81
    circle_radius: float = 2.2
    circle_samples: int = 120


CFG = StoryConfig()
DELTA = 1.0 / (2 ** (CFG.bits - 1))

# 2->2->2 network used in the brief figure generation notebook.
W1 = np.array([[0.8, 0.3], [-0.2, 0.9]], dtype=np.float64)
b1 = np.array([0.1, -0.05], dtype=np.float64)
W2 = np.array([[0.7, -0.4], [0.5, 0.6]], dtype=np.float64)
b2 = np.array([0.0, 0.0], dtype=np.float64)


def quantize_matrix(W: np.ndarray, delta: float) -> np.ndarray:
    return np.round(W / delta) * delta


W1_Q = quantize_matrix(W1, DELTA)
W2_Q = quantize_matrix(W2, DELTA)
E1 = W1_Q - W1
E2 = W2_Q - W2


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def forward_two_layer(points: np.ndarray, W1_: np.ndarray, W2_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return hidden pre-activations and output for batched points."""
    z1 = points @ W1_.T + b1
    a1 = relu(z1)
    z2 = a1 @ W2_.T + b2
    return z1, z2


def classify_sign_pattern(points: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Encode sign pattern as integer in [0, 3] for 2 hidden units."""
    z = points @ W.T + b1
    bits = (z > 0.0).astype(np.int32)
    return bits[:, 0] * 2 + bits[:, 1]


def topological_mask(points: np.ndarray) -> np.ndarray:
    return classify_sign_pattern(points, W1) != classify_sign_pattern(points, W1_Q)


def hyperplane_segment(w: np.ndarray, b: float, extent: float, n: int = 300) -> np.ndarray:
    """Sample a finite segment of w[0]x + w[1]y + b = 0 inside the view box."""
    if abs(w[1]) > abs(w[0]):
        xs = np.linspace(-extent, extent, n)
        ys = -(w[0] * xs + b) / w[1]
    else:
        ys = np.linspace(-extent, extent, n)
        xs = -(w[1] * ys + b) / w[0]

    mask = (xs >= -extent) & (xs <= extent) & (ys >= -extent) & (ys <= extent)
    return np.column_stack([xs[mask], ys[mask]])


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


def build_input_grid(extent: float, n_lines: int = 13, samples_per_line: int = 160) -> list[np.ndarray]:
    values = np.linspace(-extent, extent, n_lines)
    dense = np.linspace(-extent, extent, samples_per_line)
    lines: list[np.ndarray] = []
    for x in values:
        lines.append(np.column_stack([np.full_like(dense, x), dense]))
    for y in values:
        lines.append(np.column_stack([dense, np.full_like(dense, y)]))
    return lines


def map_through_network(points: np.ndarray, W1_: np.ndarray, W2_: np.ndarray) -> np.ndarray:
    _, out = forward_two_layer(points, W1_, W2_)
    return out


def fit_affine_transport(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine map y ~= x @ linear + offset."""
    augmented = np.column_stack([source, np.ones(source.shape[0])])
    sol, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    linear = sol[:2, :]
    offset = sol[2, :]
    return linear, offset


def apply_affine_transport(points: np.ndarray, linear: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return points @ linear + offset


def oracle_local_corrected_output(points: np.ndarray) -> np.ndarray:
    """Apply pre-ReLU local correction at both layers (exact in this toy setup)."""
    z1_q = points @ W1_Q.T + b1
    c1 = -(points @ E1.T)
    z1_corr = z1_q + c1
    a1_corr = relu(z1_corr)

    z2_q = a1_corr @ W2_Q.T + b2
    c2 = -(a1_corr @ E2.T)
    return z2_q + c2


def sample_circle(radius: float, n: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def make_panel_title(text: str) -> Text:
    return Text(text, font_size=24)


class OpeningScene(Scene):
    def construct(self) -> None:
        title = Text("Geometric Intuition for Weight Quantization", font_size=44)
        subtitle = Text("2->2->2 ReLU network, 4-bit weight-only quantization", font_size=22)
        subtitle.next_to(title, DOWN, buff=0.35)

        quant_eq = Text(
            "W_q = round(W/Delta) * Delta,  Delta = 2^(-(b-1))",
            font_size=24,
        )
        quant_eq.next_to(subtitle, DOWN, buff=0.9)

        piecewise_eq = Text(
            "f(x) = A_sigma(x) x + b_sigma(x)",
            font_size=26,
        )
        piecewise_eq.next_to(quant_eq, DOWN, buff=0.8)

        self.play(Write(title), FadeIn(subtitle, shift=UP * 0.15), run_time=1.8)
        self.play(Write(quant_eq), run_time=1.2)
        self.play(Write(piecewise_eq), run_time=1.2)
        self.wait(0.8)
        self.play(FadeOut(VGroup(title, subtitle, quant_eq, piecewise_eq)))


class PartitionShiftScene(Scene):
    def construct(self) -> None:
        title = Text("Partition Shift: Topological Distortion", font_size=34).to_edge(UP)
        plane = NumberPlane(
            x_range=[-CFG.extent, CFG.extent, 1.0],
            y_range=[-CFG.extent, CFG.extent, 1.0],
            background_line_style={"stroke_color": GREY_B, "stroke_opacity": 0.35, "stroke_width": 1},
        ).scale(0.95)

        float_lines = VGroup()
        quant_lines = VGroup()
        colors = [BLUE_D, GREEN_D]

        for idx, color in enumerate(colors):
            float_pts = hyperplane_segment(W1[idx], b1[idx], CFG.extent)
            quant_pts = hyperplane_segment(W1_Q[idx], b1[idx], CFG.extent)
            float_lines.add(polyline_from_points(plane, float_pts, color=color, stroke_width=5))
            quant_lines.add(
                polyline_from_points(plane, quant_pts, color=color, stroke_width=4, dashed=True, opacity=0.95)
            )

        xs = np.linspace(-CFG.extent, CFG.extent, CFG.grid_resolution)
        ys = np.linspace(-CFG.extent, CFG.extent, CFG.grid_resolution)
        xx, yy = np.meshgrid(xs, ys)
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        changed = grid[topological_mask(grid)]
        if changed.shape[0] > 220:
            step = max(1, changed.shape[0] // 220)
            changed = changed[::step]

        changed_dots = VGroup(
            *[
                Dot(plane.c2p(float(x), float(y)), radius=0.03, color=RED_D)
                for x, y in changed
            ]
        )

        changed_ratio = 100.0 * topological_mask(grid).mean()

        metric_text = Text("metric: A_sigma -> A_sigma + E_sigma", font_size=24)
        topo_text = Text("topological: sigma(x;W) != sigma(x;W_q)", font_size=24)
        stat_text = Text(f"Changed region area (grid estimate): {changed_ratio:.1f}%", font_size=22)

        metric_text.to_edge(DOWN).shift(UP * 1.0)
        topo_text.next_to(metric_text, DOWN, buff=0.35)
        stat_text.next_to(topo_text, DOWN, buff=0.35)

        legend_float = Text("solid = float boundaries", font_size=20, color=WHITE)
        legend_quant = Text("dashed = quantized boundaries", font_size=20, color=WHITE)
        legend = VGroup(legend_float, legend_quant).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.to_corner(UP + RIGHT).shift(LEFT * 0.5 + DOWN * 0.6)

        self.play(Write(title), Create(plane), run_time=1.5)
        self.play(Create(float_lines), run_time=1.2)
        self.play(Create(quant_lines), FadeIn(legend), run_time=1.2)
        self.play(FadeIn(changed_dots, lag_ratio=0.02), run_time=1.6)
        self.play(Write(metric_text), Write(topo_text), FadeIn(stat_text), run_time=1.4)
        self.wait(1.0)


class GridDistortionScene(Scene):
    def construct(self) -> None:
        title = Text("Metric Distortion: Wrong Affine Map in the Same Region", font_size=30).to_edge(UP)

        axes_in = Axes(
            x_range=[-CFG.extent, CFG.extent, 1],
            y_range=[-CFG.extent, CFG.extent, 1],
            axis_config={"stroke_width": 2},
            x_length=4.0,
            y_length=4.0,
            tips=False,
        )
        axes_float = axes_in.copy()
        axes_quant = axes_in.copy()

        panel_group = VGroup(axes_in, axes_float, axes_quant).arrange(RIGHT, buff=0.85).shift(DOWN * 0.25)

        input_label = make_panel_title("Input")
        float_label = make_panel_title("Float output")
        quant_label = make_panel_title("Quantized output")
        input_label.next_to(axes_in, UP, buff=0.25)
        float_label.next_to(axes_float, UP, buff=0.25)
        quant_label.next_to(axes_quant, UP, buff=0.25)

        grid_lines = build_input_grid(extent=CFG.extent, n_lines=11, samples_per_line=140)

        input_grid = VGroup(
            *[polyline_from_points(axes_in, line, color=GREY_B, stroke_width=1.3, opacity=0.9) for line in grid_lines]
        )
        float_grid = VGroup(
            *[
                polyline_from_points(
                    axes_float,
                    map_through_network(line, W1, W2),
                    color=BLUE_D,
                    stroke_width=1.8,
                    opacity=0.95,
                )
                for line in grid_lines
            ]
        )
        quant_grid = VGroup(
            *[
                polyline_from_points(
                    axes_quant,
                    map_through_network(line, W1_Q, W2_Q),
                    color=ORANGE,
                    stroke_width=1.8,
                    opacity=0.95,
                )
                for line in grid_lines
            ]
        )
        float_overlay = VGroup(
            *[
                polyline_from_points(
                    axes_quant,
                    map_through_network(line, W1, W2),
                    color=BLUE_D,
                    stroke_width=1.5,
                    opacity=0.35,
                )
                for line in grid_lines
            ]
        )

        circle = sample_circle(CFG.circle_radius, 28)
        y_float = map_through_network(circle, W1, W2)
        y_quant = map_through_network(circle, W1_Q, W2_Q)
        topological = topological_mask(circle)

        arrows = VGroup()
        for yf, yq, is_topological in zip(y_float, y_quant, topological, strict=True):
            if np.linalg.norm(yq - yf) < 1e-6:
                continue
            color = RED_D if is_topological else GREEN_D
            arrows.add(
                Arrow(
                    axes_quant.c2p(float(yf[0]), float(yf[1])),
                    axes_quant.c2p(float(yq[0]), float(yq[1])),
                    buff=0.0,
                    stroke_width=2.2,
                    max_tip_length_to_length_ratio=0.2,
                    color=color,
                )
            )

        metric_caption = Text("green arrows: metric error", font_size=20, color=GREEN_D)
        topo_caption = Text("red arrows: topological error", font_size=20, color=RED_D)
        captions = VGroup(metric_caption, topo_caption).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        captions.next_to(panel_group, DOWN, buff=0.35)

        self.play(Write(title), run_time=1.0)
        self.play(Create(panel_group), FadeIn(VGroup(input_label, float_label, quant_label)), run_time=1.2)
        self.play(Create(input_grid), run_time=1.2)
        self.play(Create(float_grid), run_time=1.2)
        self.play(Create(float_overlay), Create(quant_grid), run_time=1.4)
        self.play(Create(arrows, lag_ratio=0.06), FadeIn(captions), run_time=1.8)
        self.wait(1.0)


class PartitionToTransportScene(Scene):
    """Single-scene bridge: partition shift -> affine transport -> topological residual."""

    def construct(self) -> None:
        title = Text("From Partition Shift to Affine Transport Correction", font_size=32).to_edge(UP)

        input_plane = NumberPlane(
            x_range=[-CFG.extent, CFG.extent, 1.0],
            y_range=[-CFG.extent, CFG.extent, 1.0],
            background_line_style={"stroke_color": GREY_B, "stroke_opacity": 0.35, "stroke_width": 1},
        ).scale(0.58).shift(LEFT * 3.55 + DOWN * 0.25)

        output_axes = Axes(
            x_range=[-2.8, 2.8, 1],
            y_range=[-2.8, 2.8, 1],
            axis_config={"stroke_width": 2},
            x_length=5.1,
            y_length=5.1,
            tips=False,
        ).shift(RIGHT * 2.55 + DOWN * 0.25)

        input_label = Text("Input partition", font_size=20).next_to(input_plane, UP, buff=0.18)
        output_label = Text("Output space comparison", font_size=20).next_to(output_axes, UP, buff=0.18)

        float_lines = VGroup()
        quant_lines = VGroup()
        for idx, color in enumerate([BLUE_D, GREEN_D]):
            float_pts = hyperplane_segment(W1[idx], b1[idx], CFG.extent)
            quant_pts = hyperplane_segment(W1_Q[idx], b1[idx], CFG.extent)
            float_lines.add(polyline_from_points(input_plane, float_pts, color=color, stroke_width=3.8))
            quant_lines.add(
                polyline_from_points(input_plane, quant_pts, color=color, stroke_width=3.2, dashed=True, opacity=0.95)
            )

        xs = np.linspace(-CFG.extent, CFG.extent, CFG.grid_resolution)
        ys = np.linspace(-CFG.extent, CFG.extent, CFG.grid_resolution)
        xx, yy = np.meshgrid(xs, ys)
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        changed_mask = topological_mask(grid)
        changed = grid[changed_mask]
        if changed.shape[0] > 150:
            changed = changed[:: max(1, changed.shape[0] // 150)]
        changed_dots = VGroup(
            *[Dot(input_plane.c2p(float(x), float(y)), radius=0.02, color=RED_D) for x, y in changed]
        )
        changed_ratio = 100.0 * changed_mask.mean()

        grid_lines = build_input_grid(extent=CFG.extent, n_lines=11, samples_per_line=120)
        float_grid = VGroup(
            *[
                polyline_from_points(
                    output_axes,
                    map_through_network(line, W1, W2),
                    color=BLUE_D,
                    stroke_width=1.8,
                    opacity=0.8,
                )
                for line in grid_lines
            ]
        )
        quant_grid = VGroup(
            *[
                polyline_from_points(
                    output_axes,
                    map_through_network(line, W1_Q, W2_Q),
                    color=ORANGE,
                    stroke_width=1.8,
                    opacity=0.8,
                )
                for line in grid_lines
            ]
        )

        circle = sample_circle(CFG.circle_radius, 120)
        y_float = map_through_network(circle, W1, W2)
        y_quant = map_through_network(circle, W1_Q, W2_Q)
        topo = topological_mask(circle)
        metric = ~topo

        curve_float = polyline_from_points(output_axes, y_float, color=BLUE_D, stroke_width=4.8, close_path=True)
        curve_quant = polyline_from_points(output_axes, y_quant, color=ORANGE, stroke_width=4.8, close_path=True)

        error_arrows = VGroup()
        for idx in range(0, len(circle), 5):
            yf, yq = y_float[idx], y_quant[idx]
            if np.linalg.norm(yq - yf) < 1e-6:
                continue
            color = RED_D if topo[idx] else GREEN_D
            error_arrows.add(
                Arrow(
                    output_axes.c2p(float(yf[0]), float(yf[1])),
                    output_axes.c2p(float(yq[0]), float(yq[1])),
                    buff=0.0,
                    stroke_width=2.2,
                    max_tip_length_to_length_ratio=0.2,
                    color=color,
                )
            )

        linear, offset = fit_affine_transport(y_quant[metric], y_float[metric])
        y_transport = apply_affine_transport(y_quant, linear, offset)
        curve_transport = polyline_from_points(output_axes, y_transport, color=GREEN_D, stroke_width=4.8, close_path=True)

        topo_residual_arrows = VGroup()
        for idx in np.where(topo)[0][::2]:
            yf, yt = y_float[idx], y_transport[idx]
            if np.linalg.norm(yt - yf) < 1e-6:
                continue
            topo_residual_arrows.add(
                Arrow(
                    output_axes.c2p(float(yt[0]), float(yt[1])),
                    output_axes.c2p(float(yf[0]), float(yf[1])),
                    buff=0.0,
                    stroke_width=2.3,
                    max_tip_length_to_length_ratio=0.2,
                    color=RED_D,
                )
            )

        metric_err_before = float(np.mean(np.linalg.norm(y_quant[metric] - y_float[metric], axis=1)))
        metric_err_after = float(np.mean(np.linalg.norm(y_transport[metric] - y_float[metric], axis=1)))
        topo_err_after = float(np.mean(np.linalg.norm(y_transport[topo] - y_float[topo], axis=1)))

        transition_text = Text(
            "Boundaries are straight here, but correction lives in output/activation space.",
            font_size=18,
        ).to_edge(DOWN).shift(UP * 1.0)

        transport_eq_1 = Text("Within one region sigma:  y_q ~= T_sigma y_f + t_sigma", font_size=18)
        transport_eq_2 = Text("Inverse transport correction:  y_f ~= T_sigma^{-1}(y_q - t_sigma)", font_size=18)
        transport_eq = VGroup(transport_eq_1, transport_eq_2).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        transport_eq.to_edge(DOWN).shift(UP * 0.95)

        stats = VGroup(
            Text(f"changed partition area: {changed_ratio:.1f}%", font_size=16, color=RED_D),
            Text(f"metric mean error: {metric_err_before:.3f} -> {metric_err_after:.3f}", font_size=16, color=GREEN_D),
            Text(f"topological mean error after best affine map: {topo_err_after:.3f}", font_size=16, color=RED_D),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        stats.next_to(transport_eq, DOWN, buff=0.12)

        legend = VGroup(
            Text("blue: float", font_size=18, color=BLUE_D),
            Text("orange: quantized", font_size=18, color=ORANGE),
            Text("green: affine-transport corrected", font_size=18, color=GREEN_D),
            Text("red arrows: topological residual (not linearly correctable)", font_size=18, color=RED_D),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        legend.to_corner(UP + RIGHT).shift(LEFT * 0.3 + DOWN * 0.55)

        self.play(Write(title), run_time=1.0)
        self.play(Create(input_plane), FadeIn(input_label), run_time=1.0)
        self.play(Create(float_lines), Create(quant_lines), run_time=1.2)
        self.play(FadeIn(changed_dots, lag_ratio=0.02), run_time=1.2)
        self.play(FadeIn(transition_text), run_time=0.9)

        self.play(Create(output_axes), FadeIn(output_label), FadeIn(legend[:2]), run_time=1.1)
        self.play(Create(float_grid), Create(quant_grid), run_time=1.5)
        self.play(Create(curve_float), Create(curve_quant), run_time=1.2)
        self.play(Create(error_arrows, lag_ratio=0.08), run_time=1.4)

        self.play(FadeOut(transition_text), FadeIn(transport_eq), FadeIn(stats[0]), run_time=0.9)
        self.play(FadeOut(curve_quant), FadeOut(error_arrows), run_time=0.8)
        self.play(Create(curve_transport), FadeIn(VGroup(legend[2], stats[1])), run_time=1.1)
        self.play(Create(topo_residual_arrows), FadeIn(VGroup(legend[3], stats[2])), run_time=1.2)
        self.wait(1.0)


class ActivationSpaceComparisonScene(Scene):
    """Why L2 in activation space conflates chart mismatch with geometric error.

    Single panel, progressive reveal:
    1. Float weights create one coordinate grid (blue)
    2. Quantized weights create a different grid (orange) — same input, different stretch/rotation
    3. L2 error arrows show what the loss function actually penalises
    4. Chart alignment collapses the grids together — the "error" was entirely chart mismatch
    """

    def construct(self) -> None:
        title = Text(
            "What Does Subtracting Activations Actually Measure?",
            font_size=30,
        ).to_edge(UP)

        # Single large axes — pre-ReLU activation space (z = x W^T + b)
        axes = Axes(
            x_range=[-3.5, 3.5, 1],
            y_range=[-3.5, 3.5, 1],
            axis_config={"stroke_width": 1.5, "stroke_opacity": 0.3},
            x_length=7.5,
            y_length=7.5,
            tips=False,
        ).shift(DOWN * 0.15)

        # --- Coordinate grids: input grid mapped through W₁ and W₁_q ---
        grid_lines = build_input_grid(extent=2.0, n_lines=9, samples_per_line=80)

        float_mapped = [line @ W1.T + b1 for line in grid_lines]
        quant_mapped = [line @ W1_Q.T + b1 for line in grid_lines]

        float_grid = VGroup(*[
            polyline_from_points(axes, m, color=BLUE_D, stroke_width=1.2, opacity=0.45)
            for m in float_mapped
        ])
        quant_grid = VGroup(*[
            polyline_from_points(axes, m, color=ORANGE, stroke_width=1.2, opacity=0.45)
            for m in quant_mapped
        ])

        # --- Circle manifold mapped through both linear transforms ---
        circle = sample_circle(1.8, 80)
        z_float = circle @ W1.T + b1
        z_quant = circle @ W1_Q.T + b1

        curve_float = polyline_from_points(
            axes, z_float, color=BLUE_D, stroke_width=4.5, close_path=True,
        )
        curve_quant = polyline_from_points(
            axes, z_quant, color=ORANGE, stroke_width=4.5, close_path=True,
        )

        # --- L2 error arrows (every 8th point for readability) ---
        error_arrows = VGroup()
        for idx in range(0, len(circle), 8):
            zf, zq = z_float[idx], z_quant[idx]
            if np.linalg.norm(zq - zf) < 1e-6:
                continue
            error_arrows.add(Arrow(
                axes.c2p(float(zf[0]), float(zf[1])),
                axes.c2p(float(zq[0]), float(zq[1])),
                buff=0.0, stroke_width=2.5,
                max_tip_length_to_length_ratio=0.25, color=RED_D,
            ))

        # --- Affine chart alignment (quant chart → float chart) ---
        linear, offset = fit_affine_transport(z_quant, z_float)
        z_aligned = apply_affine_transport(z_quant, linear, offset)
        max_residual = np.max(np.linalg.norm(z_aligned - z_float, axis=1))

        aligned_mapped = [
            apply_affine_transport(m, linear, offset) for m in quant_mapped
        ]
        aligned_grid = VGroup(*[
            polyline_from_points(axes, m, color=GREEN_D, stroke_width=1.2, opacity=0.45)
            for m in aligned_mapped
        ])
        curve_aligned = polyline_from_points(
            axes, z_aligned, color=GREEN_D, stroke_width=4.5, close_path=True,
        )

        # --- Minimal labels (one caption at a time, swapped between phases) ---
        legend = VGroup(
            Text("float", font_size=16, color=BLUE_D),
            Text("quantized", font_size=16, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        legend.to_corner(UP + RIGHT).shift(LEFT * 0.3 + DOWN * 0.55)

        legend_aligned = Text("chart-aligned", font_size=16, color=GREEN_D)
        legend_aligned.next_to(legend, DOWN, aligned_edge=LEFT, buff=0.1)

        cap_grids = Text(
            "Same input \u2192 different coordinate grids",
            font_size=20,
        ).to_edge(DOWN, buff=0.3)
        cap_l2 = Text(
            "\u2016z_q \u2212 z_f\u2016 penalises coordinate disagreement, not geometric error",
            font_size=19,
        ).to_edge(DOWN, buff=0.3)
        cap_align = Text(
            f"After chart alignment: residual \u2248 {max_residual:.0e} \u2014 entirely chart mismatch",
            font_size=19, color=GREEN_D,
        ).to_edge(DOWN, buff=0.3)

        # ===== Animation =====

        self.play(Write(title), Create(axes), run_time=1.2)

        # Phase 1 — float coordinate system
        self.play(Create(float_grid), Create(curve_float), FadeIn(legend[0]), run_time=1.5)
        self.wait(0.3)

        # Phase 2 — quantized overlaid: see the stretch/rotation difference
        self.play(Create(quant_grid), Create(curve_quant), FadeIn(legend[1]), run_time=1.5)
        self.play(FadeIn(cap_grids), run_time=0.5)
        self.wait(0.8)

        # Phase 3 — L2 error arrows
        self.play(FadeOut(cap_grids), run_time=0.3)
        self.play(Create(error_arrows, lag_ratio=0.1), run_time=1.2)
        self.play(FadeIn(cap_l2), run_time=0.5)
        self.wait(0.8)

        # Phase 4 — chart alignment: orange morphs to green, overlapping blue
        self.play(FadeOut(cap_l2), FadeOut(error_arrows), run_time=0.4)
        self.play(
            Transform(quant_grid, aligned_grid),
            Transform(curve_quant, curve_aligned),
            FadeOut(legend[1]),
            FadeIn(legend_aligned),
            run_time=2.0,
        )
        self.play(FadeIn(cap_align), run_time=0.5)
        self.wait(1.5)


class ErrorDecompositionScene(Scene):
    def construct(self) -> None:
        title = Text("Error Decomposition on a Manifold", font_size=34).to_edge(UP)

        axes = Axes(
            x_range=[-2.8, 2.8, 1],
            y_range=[-2.8, 2.8, 1],
            axis_config={"stroke_width": 2},
            x_length=7.2,
            y_length=7.2,
            tips=False,
        ).shift(DOWN * 0.2)

        circle = sample_circle(CFG.circle_radius, CFG.circle_samples)
        y_float = map_through_network(circle, W1, W2)
        y_quant = map_through_network(circle, W1_Q, W2_Q)

        topo = topological_mask(circle)
        metric = ~topo
        errors = y_quant - y_float
        energy = np.sum(np.linalg.norm(errors, axis=1) ** 2)
        metric_energy = np.sum(np.linalg.norm(errors[metric], axis=1) ** 2)
        topo_energy = np.sum(np.linalg.norm(errors[topo], axis=1) ** 2)

        metric_pct = 100.0 * metric_energy / max(energy, 1e-9)
        topo_pct = 100.0 * topo_energy / max(energy, 1e-9)

        curve_float = polyline_from_points(axes, y_float, color=BLUE_D, stroke_width=5, close_path=True)
        curve_quant = polyline_from_points(axes, y_quant, color=ORANGE, stroke_width=5, close_path=True)

        arrows = VGroup()
        for idx in range(0, len(circle), 4):
            color = RED_D if topo[idx] else GREEN_D
            yf, yq = y_float[idx], y_quant[idx]
            if np.linalg.norm(yq - yf) < 1e-6:
                continue
            arrows.add(
                Arrow(
                    axes.c2p(float(yf[0]), float(yf[1])),
                    axes.c2p(float(yq[0]), float(yq[1])),
                    buff=0.0,
                    stroke_width=2.6,
                    max_tip_length_to_length_ratio=0.2,
                    color=color,
                )
            )

        eq = Text("e(x) = f_q(x) - f(x) = e_metric(x) + e_topo(x)", font_size=24)
        eq.to_edge(DOWN).shift(UP * 1.1)

        stat_1 = Text(f"metric energy: {metric_pct:.1f}%", font_size=22, color=GREEN_D)
        stat_2 = Text(f"topological energy: {topo_pct:.1f}%", font_size=22, color=RED_D)
        stats = VGroup(stat_1, stat_2).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        stats.next_to(eq, DOWN, buff=0.25)

        legend = VGroup(
            Text("blue: float manifold", font_size=20, color=BLUE_D),
            Text("orange: quantized manifold", font_size=20, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend.to_corner(UP + LEFT).shift(RIGHT * 0.4 + DOWN * 0.6)

        self.play(Write(title), Create(axes), FadeIn(legend), run_time=1.3)
        self.play(Create(curve_float), run_time=1.3)
        self.play(Create(curve_quant), run_time=1.3)
        self.play(Create(arrows, lag_ratio=0.08), run_time=1.4)
        self.play(Write(eq), FadeIn(stats), run_time=1.2)
        self.wait(1.0)


class CorrectionCascadeScene(Scene):
    def construct(self) -> None:
        title = Text("Layer-wise Correction as Inverse Distortion", font_size=30).to_edge(UP)

        axes = Axes(
            x_range=[-2.8, 2.8, 1],
            y_range=[-2.8, 2.8, 1],
            axis_config={"stroke_width": 2},
            x_length=7.2,
            y_length=7.2,
            tips=False,
        ).shift(DOWN * 0.3)

        circle = sample_circle(CFG.circle_radius, CFG.circle_samples)
        y_float = map_through_network(circle, W1, W2)
        y_quant = map_through_network(circle, W1_Q, W2_Q)
        y_corrected = oracle_local_corrected_output(circle)

        max_err_before = float(np.max(np.linalg.norm(y_quant - y_float, axis=1)))
        max_err_after = float(np.max(np.linalg.norm(y_corrected - y_float, axis=1)))

        curve_float = polyline_from_points(axes, y_float, color=BLUE_D, stroke_width=5, close_path=True)
        curve_quant = polyline_from_points(axes, y_quant, color=ORANGE, stroke_width=5, close_path=True)
        curve_corrected = polyline_from_points(axes, y_corrected, color=GREEN_D, stroke_width=5, close_path=True)

        eq_main = Text("C_L = -E_L a_L - W_L epsilon_{L-1}", font_size=26)
        eq_local = Text("toy 2-layer case: c_local = -E_L a_L is already exact", font_size=20)
        eq_main.next_to(title, DOWN, buff=0.35)
        eq_local.next_to(eq_main, DOWN, buff=0.25)

        err_text = VGroup(
            Text(f"max error before: {max_err_before:.3f}", font_size=22, color=ORANGE),
            Text(f"max error after: {max_err_after:.3e}", font_size=22, color=GREEN_D),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        err_text.next_to(axes, DOWN, buff=0.35)

        legend = VGroup(
            Text("blue: float", font_size=20, color=BLUE_D),
            Text("orange: quantized", font_size=20, color=ORANGE),
            Text("green: corrected", font_size=20, color=GREEN_D),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend.to_corner(UP + LEFT).shift(RIGHT * 0.4 + DOWN * 0.65)

        self.play(Write(title), Write(eq_main), Write(eq_local), run_time=1.8)
        self.play(Create(axes), FadeIn(legend), run_time=1.0)
        self.play(Create(curve_float), run_time=1.0)
        self.play(Create(curve_quant), FadeIn(err_text[0]), run_time=1.1)
        self.play(FadeOut(curve_quant), Create(curve_corrected), FadeIn(err_text[1]), run_time=1.2)
        self.wait(1.0)


class SequentialCorrectionScene(Scene):
    """Visualize c_local cascading perfectly through 4 layers.

    Uses 2-bit quantization (delta=0.5) so the error is ~10% of the manifold span
    and clearly visible.  Four panels show pre-ReLU manifolds at layers 1–4.
    Blue = float, Orange = uncorrected quant, Green = corrected (sequential c_local).
    """

    def construct(self) -> None:
        title = Text(
            "c_local Cascade: Exact at Every Layer",
            font_size=28,
        ).to_edge(UP)

        # --- 4-layer 2→2 network, 2-bit quantization for visible error ---
        delta = 0.5
        Ws = [
            np.array([[0.8, 0.3], [-0.2, 0.9]]),
            np.array([[0.7, -0.4], [0.5, 0.6]]),
            np.array([[0.9, -0.1], [0.4, 0.8]]),
            np.array([[0.6, 0.5], [-0.3, 0.7]]),
        ]
        bs = [
            np.array([0.1, -0.05]),
            np.array([0.0, 0.0]),
            np.array([0.05, -0.1]),
            np.array([0.0, 0.0]),
        ]
        Ws_q = [quantize_matrix(W, delta) for W in Ws]
        Es = [Wq - W for W, Wq in zip(Ws, Ws_q)]
        n_layers = len(Ws)

        circle = sample_circle(1.5, 80)

        # --- Compute pre-ReLU manifolds for all three paths ---
        float_pre: list[np.ndarray] = []
        quant_pre: list[np.ndarray] = []
        corrected_pre: list[np.ndarray] = []
        quant_errs: list[float] = []
        corr_errs: list[float] = []

        a_f, a_q, a_c = circle.copy(), circle.copy(), circle.copy()
        for W, Wq, E, b in zip(Ws, Ws_q, Es, bs):
            zf = a_f @ W.T + b
            zq = a_q @ Wq.T + b
            zc_raw = a_c @ Wq.T + b
            zc = zc_raw + (-a_c @ E.T)  # z_q + c_local

            float_pre.append(zf)
            quant_pre.append(zq)
            corrected_pre.append(zc)
            quant_errs.append(float(np.mean(np.linalg.norm(zq - zf, axis=1))))
            corr_errs.append(float(np.max(np.abs(zc - zf))))

            a_f = relu(zf)
            a_q = relu(zq)
            a_c = relu(zc)

        # --- Build per-layer panels with fitted axes ---
        axes_list: list[Axes] = []
        for i in range(n_layers):
            all_pts = np.vstack([float_pre[i], quant_pre[i]])
            xmin, xmax = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
            ymin, ymax = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
            span = max(xmax - xmin, ymax - ymin, 0.5)
            pad = 0.15 * span
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            half = span / 2 + pad

            axes_list.append(
                Axes(
                    x_range=[cx - half, cx + half, half],
                    y_range=[cy - half, cy + half, half],
                    axis_config={"stroke_width": 1, "stroke_opacity": 0.25},
                    x_length=3.0,
                    y_length=3.0,
                    tips=False,
                )
            )

        panel_group = VGroup(*axes_list).arrange(RIGHT, buff=0.35).shift(DOWN * 0.1)

        layer_labels = VGroup(
            *[
                Text(f"Layer {i + 1}", font_size=16).next_to(axes_list[i], UP, buff=0.12)
                for i in range(n_layers)
            ]
        )

        # --- Curves ---
        float_curves = [
            polyline_from_points(axes_list[i], float_pre[i], color=BLUE_D, stroke_width=3.0, close_path=True)
            for i in range(n_layers)
        ]
        quant_curves = [
            polyline_from_points(axes_list[i], quant_pre[i], color=ORANGE, stroke_width=3.0, close_path=True)
            for i in range(n_layers)
        ]
        corrected_curves = [
            polyline_from_points(axes_list[i], corrected_pre[i], color=GREEN_D, stroke_width=3.0, close_path=True)
            for i in range(n_layers)
        ]

        # --- Error labels below each panel ---
        err_labels = VGroup(
            *[
                VGroup(
                    Text(f"err: {quant_errs[i]:.3f}", font_size=13, color=ORANGE),
                    Text(f"err: {corr_errs[i]:.0e}", font_size=13, color=GREEN_D),
                )
                .arrange(DOWN, aligned_edge=LEFT, buff=0.04)
                .next_to(axes_list[i], DOWN, buff=0.12)
                for i in range(n_layers)
            ]
        )

        legend = VGroup(
            Text("float", font_size=14, color=BLUE_D),
            Text("quant (no correction)", font_size=14, color=ORANGE),
            Text("sequential c_local", font_size=14, color=GREEN_D),
        ).arrange(RIGHT, buff=0.5)
        legend.to_edge(DOWN, buff=0.15)

        # ===== Animation: progressive layer-by-layer fill =====

        self.play(Write(title), run_time=0.8)
        self.play(
            *[Create(ax) for ax in axes_list],
            FadeIn(layer_labels),
            FadeIn(legend),
            run_time=1.0,
        )

        for i in range(n_layers):
            # Float manifold (blue)
            self.play(Create(float_curves[i]), run_time=0.5)
            # Quantized manifold (orange) — visible error
            self.play(Create(quant_curves[i]), run_time=0.5)
            # Correction: orange morphs to green, snapping onto blue
            self.play(
                Transform(quant_curves[i], corrected_curves[i]),
                FadeIn(err_labels[i]),
                run_time=0.8,
            )
            self.wait(0.2)

        self.wait(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Narrative arc: 5-scene sequence explaining weight quantization geometry
# Uses 2-bit quantization (delta=0.5) for visible errors (~10% of span)
# ═══════════════════════════════════════════════════════════════════════════════

NARR_DELTA = 0.5
NARR_WS = [
    np.array([[0.8, 0.3], [-0.2, 0.9]]),
    np.array([[0.7, -0.4], [0.5, 0.6]]),
    np.array([[0.9, -0.1], [0.4, 0.8]]),
]
NARR_BS = [
    np.array([0.1, -0.05]),
    np.array([0.0, 0.0]),
    np.array([0.05, -0.1]),
]
NARR_WS_Q = [quantize_matrix(W, NARR_DELTA) for W in NARR_WS]
NARR_ES = [Wq - W for W, Wq in zip(NARR_WS, NARR_WS_Q)]


def _narr_fitted_axes(
    points_list: list[np.ndarray],
    size: float = 6.0,
    pad_frac: float = 0.15,
) -> Axes:
    """Build axes fitted to the bounding box of multiple point arrays."""
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


class LayerProgressionScene(Scene):
    """Scene 1: What a ReLU network does to data layer by layer.

    Single axes, a circle manifold gets progressively transformed:
    W₁ → ReLU → W₂ → ReLU → W₃ → ReLU.  Each step animates via Transform.
    """

    def construct(self) -> None:
        title = Text(
            "What a ReLU Network Does to Data",
            font_size=32,
        ).to_edge(UP)

        circle = sample_circle(1.5, 100)

        # Pre-compute all intermediate manifolds (float only)
        manifolds = [circle.copy()]
        a = circle.copy()
        for W, b in zip(NARR_WS, NARR_BS):
            z = a @ W.T + b
            manifolds.append(z.copy())  # pre-ReLU
            a = relu(z)
            manifolds.append(a.copy())  # post-ReLU

        # We'll show: input → W₁ → ReLU → W₂ → ReLU → W₃ → ReLU
        step_labels = [
            "input circle",
            "after W₁ (affine stretch)",
            "after ReLU (fold)",
            "after W₂ (affine stretch)",
            "after ReLU (fold)",
            "after W₃ (affine stretch)",
            "after ReLU (fold)",
        ]

        # Build fitted axes for each step
        axes_per_step = [
            _narr_fitted_axes([m], size=6.5) for m in manifolds
        ]

        # Start with input circle
        ax = axes_per_step[0].shift(DOWN * 0.15)
        curve = polyline_from_points(
            ax, manifolds[0], color=BLUE_D, stroke_width=4, close_path=True,
        )
        caption = Text(step_labels[0], font_size=22).to_edge(DOWN, buff=0.35)

        self.play(Write(title), run_time=0.8)
        self.play(Create(ax), Create(curve), FadeIn(caption), run_time=1.2)
        self.wait(0.5)

        for i in range(1, len(manifolds)):
            new_ax = axes_per_step[i].shift(DOWN * 0.15)
            new_curve = polyline_from_points(
                new_ax, manifolds[i], color=BLUE_D, stroke_width=4, close_path=True,
            )
            new_caption = Text(step_labels[i], font_size=22).to_edge(DOWN, buff=0.35)

            self.play(
                Transform(ax, new_ax),
                Transform(curve, new_curve),
                Transform(caption, new_caption),
                run_time=1.0 if "ReLU" in step_labels[i] else 1.2,
            )
            self.wait(0.4)

        self.wait(1.0)


class QuantizationComparisonScene(Scene):
    """Scene 2: What happens when you add quantization.

    Two curves transform simultaneously through the same layer steps.
    Blue = float, Orange = quantized.  The divergence grows at each step.
    """

    def construct(self) -> None:
        title = Text(
            "Adding Quantization: Two Paths Diverge",
            font_size=32,
        ).to_edge(UP)

        circle = sample_circle(1.5, 100)

        # Pre-compute float and quant manifolds at each step
        f_manifolds = [circle.copy()]
        q_manifolds = [circle.copy()]
        a_f, a_q = circle.copy(), circle.copy()
        for W, Wq, b in zip(NARR_WS, NARR_WS_Q, NARR_BS):
            zf = a_f @ W.T + b
            zq = a_q @ Wq.T + b
            f_manifolds.append(zf)
            q_manifolds.append(zq)
            a_f = relu(zf)
            a_q = relu(zq)
            f_manifolds.append(a_f.copy())
            q_manifolds.append(a_q.copy())

        step_labels = [
            "input (shared)",
            "after W₁",
            "after ReLU",
            "after W₂",
            "after ReLU",
            "after W₃",
            "after ReLU",
        ]

        # Fitted axes using BOTH float and quant points
        axes_per_step = [
            _narr_fitted_axes([f_manifolds[i], q_manifolds[i]], size=6.5)
            for i in range(len(f_manifolds))
        ]

        legend = VGroup(
            Text("float", font_size=16, color=BLUE_D),
            Text("quantized", font_size=16, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        legend.to_corner(UP + RIGHT).shift(LEFT * 0.3 + DOWN * 0.55)

        # Start with shared input
        ax = axes_per_step[0].shift(DOWN * 0.15)
        c_f = polyline_from_points(
            ax, f_manifolds[0], color=BLUE_D, stroke_width=4, close_path=True,
        )
        c_q = polyline_from_points(
            ax, q_manifolds[0], color=ORANGE, stroke_width=3.5, close_path=True,
        )
        caption = Text(step_labels[0], font_size=22).to_edge(DOWN, buff=0.35)

        self.play(Write(title), FadeIn(legend), run_time=0.8)
        self.play(Create(ax), Create(c_f), Create(c_q), FadeIn(caption), run_time=1.2)
        self.wait(0.4)

        for i in range(1, len(f_manifolds)):
            new_ax = axes_per_step[i].shift(DOWN * 0.15)
            new_cf = polyline_from_points(
                new_ax, f_manifolds[i], color=BLUE_D, stroke_width=4, close_path=True,
            )
            new_cq = polyline_from_points(
                new_ax, q_manifolds[i], color=ORANGE, stroke_width=3.5, close_path=True,
            )
            new_caption = Text(step_labels[i], font_size=22).to_edge(DOWN, buff=0.35)

            # Show error stat after each W step (odd indices = pre-ReLU)
            err = float(np.mean(np.linalg.norm(
                f_manifolds[i] - q_manifolds[i], axis=1,
            )))
            anims = [
                Transform(ax, new_ax),
                Transform(c_f, new_cf),
                Transform(c_q, new_cq),
                Transform(caption, new_caption),
            ]
            self.play(*anims, run_time=1.0)
            self.wait(0.3)

        self.wait(1.0)


def _narr_sign_pattern(points: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Encode per-point ReLU sign pattern as integer."""
    z = points @ W.T + b
    bits = (z > 0.0).astype(np.int32)
    return bits[:, 0] * 2 + bits[:, 1]


class CorrectionAndTopologyScene(Scene):
    """Scenes 3+4: Linear correction, and where it fails (topology).

    Phase A — Post-output correction:
      After 3-layer float vs quant, apply global affine correction.
      Points colored green (metric = same ReLU region) vs red (topological = different region).
      Show affine correction collapses green points; red points remain as residual.

    Phase B — Zoom on the residual:
      Only red (topological) points with error arrows. Caption: these crossed
      a ReLU boundary, so no single linear map can fix them.
    """

    def construct(self) -> None:
        # --- Forward pass through 3 layers (float & quant) ---
        circle = sample_circle(1.5, 200)

        a_f, a_q = circle.copy(), circle.copy()
        topo_mask = np.zeros(len(circle), dtype=bool)
        for W, Wq, b in zip(NARR_WS, NARR_WS_Q, NARR_BS):
            zf = a_f @ W.T + b
            zq = a_q @ Wq.T + b
            # Mark points where ReLU pattern differs at this layer
            pat_f = _narr_sign_pattern(a_f, W, b)
            pat_q = _narr_sign_pattern(a_q, Wq, b)
            topo_mask |= (pat_f != pat_q)
            a_f = relu(zf)
            a_q = relu(zq)

        # Final output (post last ReLU)
        y_f, y_q = a_f, a_q
        metric_mask = ~topo_mask

        n_metric = int(metric_mask.sum())
        n_topo = int(topo_mask.sum())
        metric_pct = 100.0 * n_metric / len(circle)

        # --- Phase A: full output with metric/topological coloring ---
        title_a = Text(
            "Linear Correction: What It Can and Cannot Fix",
            font_size=28,
        ).to_edge(UP)

        ax = _narr_fitted_axes([y_f, y_q], size=6.5).shift(DOWN * 0.15)

        # Dots colored by type
        dot_radius = 0.035
        float_dots = VGroup(*[
            Dot(
                ax.c2p(float(y_f[i, 0]), float(y_f[i, 1])),
                radius=dot_radius, color=BLUE_D, fill_opacity=0.5,
            )
            for i in range(len(circle))
        ])
        quant_dots = VGroup(*[
            Dot(
                ax.c2p(float(y_q[i, 0]), float(y_q[i, 1])),
                radius=dot_radius,
                color=GREEN_D if metric_mask[i] else RED_D,
                fill_opacity=0.8,
            )
            for i in range(len(circle))
        ])

        legend = VGroup(
            Text("float output", font_size=15, color=BLUE_D),
            Text(f"metric ({metric_pct:.0f}%)", font_size=15, color=GREEN_D),
            Text(f"topological ({100 - metric_pct:.0f}%)", font_size=15, color=RED_D),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        legend.to_corner(UP + RIGHT).shift(LEFT * 0.3 + DOWN * 0.55)

        cap_before = Text(
            "Quantized output: green = same ReLU regions, red = different regions",
            font_size=19,
        ).to_edge(DOWN, buff=0.3)

        self.play(Write(title_a), Create(ax), FadeIn(legend), run_time=1.0)
        self.play(FadeIn(float_dots, lag_ratio=0.01), run_time=1.0)
        self.play(FadeIn(quant_dots, lag_ratio=0.01), FadeIn(cap_before), run_time=1.2)
        self.wait(0.8)

        # --- Affine correction: fit on metric points, apply to all ---
        linear, offset = fit_affine_transport(
            y_q[metric_mask], y_f[metric_mask],
        )
        y_corrected = apply_affine_transport(y_q, linear, offset)

        metric_residual = float(np.mean(
            np.linalg.norm(y_corrected[metric_mask] - y_f[metric_mask], axis=1),
        ))
        topo_residual = float(np.mean(
            np.linalg.norm(y_corrected[topo_mask] - y_f[topo_mask], axis=1),
        )) if n_topo > 0 else 0.0

        corrected_dots = VGroup(*[
            Dot(
                ax.c2p(float(y_corrected[i, 0]), float(y_corrected[i, 1])),
                radius=dot_radius,
                color=GREEN_D if metric_mask[i] else RED_D,
                fill_opacity=0.8,
            )
            for i in range(len(circle))
        ])

        cap_after = Text(
            f"After affine correction: metric residual {metric_residual:.4f}, "
            f"topological residual {topo_residual:.3f}",
            font_size=18,
        ).to_edge(DOWN, buff=0.3)

        self.play(FadeOut(cap_before), run_time=0.3)
        self.play(
            Transform(quant_dots, corrected_dots),
            run_time=2.0,
        )
        self.play(FadeIn(cap_after), run_time=0.5)
        self.wait(1.0)

        # --- Phase B: Residual arrows on topological points ---
        topo_arrows = VGroup()
        for i in np.where(topo_mask)[0]:
            yf_i, yc_i = y_f[i], y_corrected[i]
            dist = np.linalg.norm(yc_i - yf_i)
            if dist < 1e-6:
                continue
            topo_arrows.add(Arrow(
                ax.c2p(float(yf_i[0]), float(yf_i[1])),
                ax.c2p(float(yc_i[0]), float(yc_i[1])),
                buff=0.0, stroke_width=2.5,
                max_tip_length_to_length_ratio=0.25, color=RED_D,
            ))

        cap_topo = Text(
            "Red residuals: these crossed a ReLU boundary — no single linear map can fix them",
            font_size=18, color=RED_D,
        ).to_edge(DOWN, buff=0.3)

        self.play(FadeOut(cap_after), run_time=0.3)
        self.play(Create(topo_arrows, lag_ratio=0.05), run_time=1.2)
        self.play(FadeIn(cap_topo), run_time=0.5)
        self.wait(1.5)


class FilteredLowRankScene(Scene):
    """Scene 5: Only linearly correctable points for low-rank estimation.

    Two panels side by side showing oracle correction vectors at the final layer:
      Left:  all points (metric green + topological red scatter)
      Right: metric-only points (cleaner, tighter structure)

    Each panel shows SVD principal direction as a line through the scatter.
    The insight: topological outliers distort the principal direction.
    """

    def construct(self) -> None:
        title = Text(
            "Filtering for Better Low-Rank Correction",
            font_size=28,
        ).to_edge(UP)

        # --- Forward pass: collect corrections at final layer ---
        circle = sample_circle(1.5, 200)

        a_f, a_q = circle.copy(), circle.copy()
        topo_mask = np.zeros(len(circle), dtype=bool)
        for W, Wq, b in zip(NARR_WS, NARR_WS_Q, NARR_BS):
            zf = a_f @ W.T + b
            zq = a_q @ Wq.T + b
            pat_f = _narr_sign_pattern(a_f, W, b)
            pat_q = _narr_sign_pattern(a_q, Wq, b)
            topo_mask |= (pat_f != pat_q)
            a_f = relu(zf)
            a_q = relu(zq)

        # Oracle corrections at final output (post all ReLUs)
        corrections = a_f - a_q  # what you'd need to add to quant to match float
        metric_mask = ~topo_mask

        # SVD of all corrections vs metric-only
        def svd_dir(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return top-1 SVD direction and singular values."""
            C_centered = C - C.mean(axis=0)
            U, S, Vt = np.linalg.svd(C_centered, full_matrices=False)
            return Vt[0], S

        dir_all, s_all = svd_dir(corrections)
        dir_metric, s_metric = svd_dir(corrections[metric_mask])

        # Energy in top-1 direction
        e1_all = s_all[0] ** 2 / max(np.sum(s_all**2), 1e-12) * 100
        e1_metric = s_metric[0] ** 2 / max(np.sum(s_metric**2), 1e-12) * 100

        # --- Two panels ---
        def build_panel(
            pts: np.ndarray,
            masks: np.ndarray,
            svd_direction: np.ndarray,
            panel_title: str,
            energy_pct: float,
        ) -> tuple[VGroup, Axes]:
            ax = _narr_fitted_axes([pts], size=5.0, pad_frac=0.2)

            dots = VGroup()
            for i in range(len(pts)):
                color = GREEN_D if masks[i] else RED_D
                dots.add(Dot(
                    ax.c2p(float(pts[i, 0]), float(pts[i, 1])),
                    radius=0.03, color=color, fill_opacity=0.7,
                ))

            # SVD principal direction line through centroid
            center = pts.mean(axis=0)
            span = max(pts[:, 0].max() - pts[:, 0].min(),
                       pts[:, 1].max() - pts[:, 1].min(), 0.5) * 0.45
            p1 = center - span * svd_direction
            p2 = center + span * svd_direction
            svd_line = polyline_from_points(
                ax, np.array([p1, p2]), color=WHITE, stroke_width=2.5,
            )

            label = Text(panel_title, font_size=16)
            energy_label = Text(
                f"rank-1 energy: {energy_pct:.1f}%", font_size=14,
            )
            panel = VGroup(ax, dots, svd_line)
            label.next_to(ax, UP, buff=0.1)
            energy_label.next_to(ax, DOWN, buff=0.1)
            panel.add(label, energy_label)
            return panel, ax

        panel_all, ax_all = build_panel(
            corrections, metric_mask, dir_all,
            "all points", e1_all,
        )
        panel_metric, ax_metric = build_panel(
            corrections[metric_mask],
            np.ones(int(metric_mask.sum()), dtype=bool),
            dir_metric,
            "metric-only (filtered)", e1_metric,
        )

        panels = VGroup(panel_all, panel_metric).arrange(RIGHT, buff=0.6).shift(DOWN * 0.1)

        legend = VGroup(
            Text("metric correction", font_size=14, color=GREEN_D),
            Text("topological correction", font_size=14, color=RED_D),
            Text("SVD principal direction", font_size=14, color=WHITE),
        ).arrange(RIGHT, buff=0.5)
        legend.to_edge(DOWN, buff=0.15)

        cap = Text(
            "Topological outliers distort the low-rank correction basis",
            font_size=19,
        ).to_edge(DOWN, buff=0.5)

        # ===== Animation =====
        self.play(Write(title), run_time=0.8)

        # Left panel first
        self.play(
            Create(ax_all),
            FadeIn(panel_all[3]),  # label
            run_time=0.8,
        )
        self.play(FadeIn(panel_all[1], lag_ratio=0.01), run_time=1.0)  # dots
        self.play(Create(panel_all[2]), FadeIn(panel_all[4]), run_time=0.8)  # svd line + energy
        self.wait(0.5)

        # Right panel
        self.play(
            Create(ax_metric),
            FadeIn(panel_metric[3]),
            run_time=0.8,
        )
        self.play(FadeIn(panel_metric[1], lag_ratio=0.01), run_time=1.0)
        self.play(Create(panel_metric[2]), FadeIn(panel_metric[4]), run_time=0.8)

        self.play(FadeIn(legend), FadeIn(cap), run_time=0.8)
        self.wait(1.5)


class FullNarrativeScene(Scene):
    """A convenience scene to stitch key moments into one render."""

    def construct(self) -> None:
        opening = OpeningScene.construct
        partition_to_transport = PartitionToTransportScene.construct
        activation_compare = ActivationSpaceComparisonScene.construct
        decomposition = ErrorDecompositionScene.construct
        correction = CorrectionCascadeScene.construct

        for segment in (opening, partition_to_transport, activation_compare, decomposition, correction):
            segment(self)
            self.wait(0.3)
            self.clear()
