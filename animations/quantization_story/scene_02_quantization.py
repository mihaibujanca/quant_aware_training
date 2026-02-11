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
    NARR_WS_Q,
    NARR_BS,
    as_column_vector,
    build_float_quant_progression,
    narr_global_axes,
    polyline_from_points,
    sample_circle,
)


class QuantizationComparisonScene(Scene):
    """2) Add quantization and show float vs quantized path divergence."""

    def construct(self) -> None:
        title = Text("Adding Quantization: Two Paths Diverge", font_size=32).to_edge(UP)

        circle = sample_circle(1.5, 170)
        f_manifolds, q_manifolds, labels, kinds = build_float_quant_progression(circle)

        legend = VGroup(
            Text("float", font_size=16, color=BLUE_D),
            Text("quantized", font_size=16, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        legend.to_corner(UP + RIGHT).shift(LEFT * 0.3 + DOWN * 0.55)

        ax = narr_global_axes(f_manifolds + q_manifolds, size=6.6).shift(DOWN * 0.08)
        c_f = polyline_from_points(ax, f_manifolds[0], color=BLUE_D, stroke_width=4, close_path=True)
        c_q = polyline_from_points(ax, q_manifolds[0], color=ORANGE, stroke_width=3.5, close_path=True)
        caption = Text(labels[0], font_size=20).to_edge(DOWN, buff=0.28)

        def layer_panel(layer_idx: int) -> VGroup:
            W = NARR_WS[layer_idx - 1]
            Wq = NARR_WS_Q[layer_idx - 1]
            b = as_column_vector(NARR_BS[layer_idx - 1])
            # bq = NARR_BS_Q[layer_idx - 1]

            layer_title = Text(f"Layer {layer_idx}", font_size=15)

            w_label = MathTex(fr"W_{{{layer_idx}}}=", font_size=28, color=BLUE_D)
            w_matrix = DecimalMatrix(
                W,
                element_to_mobject_config={"num_decimal_places": 2, "font_size": 32},
                h_buff=1.65,
                v_buff=0.72,
            ).scale(0.54)
            w_line = VGroup(w_label, w_matrix).arrange(RIGHT, buff=0.1, aligned_edge=UP)

            wq_label = MathTex(fr"W^q_{{{layer_idx}}}=", font_size=28, color=ORANGE)
            wq_matrix = DecimalMatrix(
                Wq,
                element_to_mobject_config={"num_decimal_places": 2, "font_size": 32},
                h_buff=1.65,
                v_buff=0.72,
            ).scale(0.54)
            wq_line = VGroup(wq_label, wq_matrix).arrange(RIGHT, buff=0.1, aligned_edge=UP)

            b_label = MathTex(fr"b_{{{layer_idx}}}=", font_size=28)
            b_vec = DecimalMatrix(
                b,
                element_to_mobject_config={"num_decimal_places": 2, "font_size": 32},
                h_buff=1.2,
                v_buff=0.72,
            ).scale(0.54)
            b_line = VGroup(b_label, b_vec).arrange(RIGHT, buff=0.1, aligned_edge=UP)

            lines = VGroup(layer_title, w_line, wq_line, b_line).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
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

        self.play(Write(title), FadeIn(legend), run_time=0.8)
        self.play(Create(ax), Create(c_f), Create(c_q), FadeIn(caption), FadeIn(param_panel), run_time=1.2)
        self.wait(0.4)

        for i in range(1, len(f_manifolds)):
            kind = kinds[i]
            new_cf = polyline_from_points(ax, f_manifolds[i], color=BLUE_D, stroke_width=4, close_path=True)
            new_cq = polyline_from_points(ax, q_manifolds[i], color=ORANGE, stroke_width=3.5, close_path=True)
            new_caption = Text(labels[i], font_size=20).to_edge(DOWN, buff=0.28)

            if kind == "affine":
                layer_idx = (i + 1) // 2
                new_panel = layer_panel(layer_idx)
                err = float(np.mean(np.linalg.norm(f_manifolds[i] - q_manifolds[i], axis=1)))
                err_text = Text(f"mean separation: {err:.3f}", font_size=16, color=ORANGE)
                err_text.to_corner(UP + LEFT).shift(RIGHT * 0.35 + DOWN * 0.55)
                self.play(
                    Transform(c_f, new_cf),
                    Transform(c_q, new_cq),
                    Transform(caption, new_caption),
                    Transform(param_panel, new_panel),
                    run_time=1.0,
                )
                self.play(FadeIn(err_text), run_time=0.3)
                self.play(FadeOut(err_text), run_time=0.25)
            else:
                pre_f = f_manifolds[i - 1]
                pre_q = q_manifolds[i - 1]

                clipped_f = pre_f[np.any(pre_f < 0.0, axis=1)]
                clipped_q = pre_q[np.any(pre_q < 0.0, axis=1)]
                if len(clipped_f) > 45:
                    clipped_f = clipped_f[:: max(1, len(clipped_f) // 45)]
                if len(clipped_q) > 45:
                    clipped_q = clipped_q[:: max(1, len(clipped_q) // 45)]

                clipped_f_dots = VGroup(
                    *[
                        Dot(ax.c2p(float(px), float(py)), radius=0.022, color=BLUE_D)
                        for px, py in clipped_f
                    ]
                )
                clipped_q_dots = VGroup(
                    *[
                        Dot(ax.c2p(float(px), float(py)), radius=0.022, color=ORANGE)
                        for px, py in clipped_q
                    ]
                )

                self.play(FadeIn(relu_guides), FadeIn(clipped_f_dots), FadeIn(clipped_q_dots), run_time=0.4)
                self.play(
                    Transform(c_f, new_cf),
                    Transform(c_q, new_cq),
                    Transform(caption, new_caption),
                    run_time=0.95,
                )
                self.play(FadeOut(clipped_f_dots), FadeOut(clipped_q_dots), FadeOut(relu_guides), run_time=0.35)

            self.wait(0.25)

        self.wait(1.0)
