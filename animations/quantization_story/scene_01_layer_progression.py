from __future__ import annotations

import numpy as np
from manim import (
    BLUE_D,
    DecimalMatrix,
    GREY_B,
    MathTex,
    ORANGE,
    RED_D,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Dot,
    FadeIn,
    FadeOut,
    Create,
    Scene,
    SurroundingRectangle,
    Text,
    Transform,
    Write,
    VGroup,
)

from .common import (
    NARR_WS,
    NARR_BS,
    as_column_vector,
    build_float_progression_steps,
    narr_global_axes,
    polyline_from_points,
    sample_circle,
)


class LayerProgressionScene(Scene):
    """1) What a simple ReLU network does through layers (explicit W then ReLU)."""

    def construct(self) -> None:
        title = Text("What a ReLU Network Does to Data", font_size=32).to_edge(UP)

        circle = sample_circle(1.5, 180)
        steps = build_float_progression_steps(circle, ws=NARR_WS)

        ax = narr_global_axes([pts for _, pts, _ in steps], size=6.6).shift(DOWN * 0.08)
        curve = polyline_from_points(ax, steps[0][1], color=BLUE_D, stroke_width=4, close_path=True)
        caption = Text(steps[0][0], font_size=20).to_edge(DOWN, buff=0.28)

        def layer_panel(layer_idx: int) -> VGroup:
            W = NARR_WS[layer_idx - 1]
            b = as_column_vector(NARR_BS[layer_idx - 1])

            layer_title = Text(f"Layer {layer_idx}", font_size=15)

            w_label = MathTex(fr"W_{{{layer_idx}}}=", font_size=28)
            w_matrix = DecimalMatrix(
                W,
                element_to_mobject_config={"num_decimal_places": 2, "font_size": 34},
                h_buff=1.8,
                v_buff=0.75,
            ).scale(0.58)
            w_line = VGroup(w_label, w_matrix).arrange(RIGHT, buff=0.1, aligned_edge=UP)

            b_label = MathTex(fr"b_{{{layer_idx}}}=", font_size=28)
            b_vec = DecimalMatrix(
                b,
                element_to_mobject_config={"num_decimal_places": 2, "font_size": 34},
                h_buff=1.2,
                v_buff=0.75,
            ).scale(0.58)
            b_line = VGroup(b_label, b_vec).arrange(RIGHT, buff=0.1, aligned_edge=UP)

            lines = VGroup(layer_title, w_line, b_line).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
            box = SurroundingRectangle(lines, buff=0.12, color=GREY_B, stroke_width=1.2)
            panel = VGroup(box, lines)
            panel.to_corner(DOWN + RIGHT, buff=0.22).shift(UP * 0.78)
            return panel

        param_panel = layer_panel(1)

        x0, x1 = ax.x_range[0], ax.x_range[1]
        y0, y1 = ax.y_range[0], ax.y_range[1]
        relu_guides = VGroup(
            polyline_from_points(ax, np.array([[x0, 0.0], [x1, 0.0]]), color=RED_D, stroke_width=3.2),
            polyline_from_points(ax, np.array([[0.0, y0], [0.0, y1]]), color=RED_D, stroke_width=3.2),
        )

        self.play(Write(title), run_time=0.7)
        self.play(Create(ax), Create(curve), FadeIn(caption), FadeIn(param_panel), run_time=1.1)
        self.wait(0.3)

        for i in range(1, len(steps)):
            label, pts, kind = steps[i]
            new_curve = polyline_from_points(
                ax,
                pts,
                color=ORANGE if kind == "affine" else BLUE_D,
                stroke_width=4,
                close_path=True,
            )
            new_caption = Text(label, font_size=20).to_edge(DOWN, buff=0.28)

            if kind == "affine":
                layer_idx = (i + 1) // 2
                new_panel = layer_panel(layer_idx)
                self.play(
                    Transform(curve, new_curve),
                    Transform(caption, new_caption),
                    Transform(param_panel, new_panel),
                    run_time=1.0,
                )
            else:
                pre_pts = steps[i - 1][1]
                clipped = pre_pts[np.any(pre_pts < 0.0, axis=1)]
                if len(clipped) > 55:
                    clipped = clipped[:: max(1, len(clipped) // 55)]

                clipped_dots = VGroup(
                    *[
                        Dot(ax.c2p(float(px), float(py)), radius=0.025, color=RED_D)
                        for px, py in clipped
                    ]
                )

                self.play(FadeIn(relu_guides), FadeIn(clipped_dots), run_time=0.45)
                self.play(
                    Transform(curve, new_curve),
                    Transform(caption, new_caption),
                    run_time=0.95,
                )
                self.play(FadeOut(clipped_dots), FadeOut(relu_guides), run_time=0.35)

            self.wait(0.2)

        self.wait(0.9)
