from __future__ import annotations

import numpy as np
from manim import (
    BLACK,
    BLUE_D,
    Create,
    Dot,
    DOWN,
    FadeIn,
    GREY_B,
    LEFT,
    Line,
    NumberLine,
    ORANGE,
    RED_D,
    RIGHT,
    Scene,
    SurroundingRectangle,
    Text,
    Transform,
    UP,
    VGroup,
    Write,
    YELLOW_D,
    Arrow,
)


class OneDErrorGrowthScene(Scene):
    """Intro-only 1D scene: how int4 quantize/dequantize works."""

    def construct(self) -> None:
        bits = 4
        qmax = (2 ** (bits - 1)) - 1
        qmin = -qmax  # practical symmetric int4: [-7, 7]
        scale = 0.1
        x_min, x_max = qmin * scale, qmax * scale

        x = 0.36
        q_cont = x / scale
        q = int(np.clip(np.round(q_cont), qmin, qmax))
        x_hat = float(q * scale)
        err = abs(x_hat - x)

        title = Text("How 4-Bit Quantization Works", font_size=34).to_edge(UP, buff=0.2)
        subtitle = Text("Map real value x to integer code q, then map back to x_hat", font_size=20).next_to(
            title, DOWN, buff=0.12
        )

        q_axis = NumberLine(
            x_range=[-8, 8, 1],
            length=9.4,
            include_numbers=True,
            include_ticks=True,
            font_size=24,
        ).shift(DOWN * 0.95)
        q_axis_label = Text("Integer code axis q", font_size=17).next_to(q_axis, UP, buff=0.1)

        x_axis = NumberLine(
            x_range=[-1.0, 1.0, 0.1],
            length=9.4,
            include_numbers=True,
            include_ticks=True,
            font_size=22,
        ).shift(DOWN * 2.35)
        x_axis_label = Text("Dequantized real axis x_hat = s * q   (s = 0.1)", font_size=17).next_to(x_axis, UP, buff=0.1)

        x_levels = VGroup(
            *[
                Line(
                    x_axis.n2p(float(v)) + DOWN * 0.06,
                    x_axis.n2p(float(v)) + UP * 0.06,
                    color=GREY_B,
                    stroke_width=2,
                    stroke_opacity=0.72,
                )
                for v in np.arange(x_min, x_max + 1e-9, scale)
            ]
        )

        q_l = Line(q_axis.n2p(qmin) + DOWN * 0.22, q_axis.n2p(qmin) + UP * 0.22, color=RED_D)
        q_r = Line(q_axis.n2p(qmax) + DOWN * 0.22, q_axis.n2p(qmax) + UP * 0.22, color=RED_D)

        x_l = Line(x_axis.n2p(x_min) + DOWN * 0.22, x_axis.n2p(x_min) + UP * 0.22, color=RED_D)
        x_r = Line(x_axis.n2p(x_max) + DOWN * 0.22, x_axis.n2p(x_max) + UP * 0.22, color=RED_D)

        legend = VGroup(
            Text("blue: float value", font_size=16, color=BLUE_D),
            Text("orange: quantized code / dequantized value", font_size=16, color=ORANGE),
            Text("yellow: quantization error |x_hat - x|", font_size=16, color=YELLOW_D),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
        legend.to_edge(LEFT, buff=0.4).set_y(1.5)

        panel_content = VGroup(
            Text("Symmetric int4 setup", font_size=17, color=RED_D),
            Text("Code range: q in [-7, 7]", font_size=16),
            Text("Quantize: q = clip(round(x/s), -7, 7)", font_size=16),
            Text("Dequantize: x_hat = s * q", font_size=16),
            Text("Example: x=0.36 -> q=4 -> x_hat=0.40", font_size=16),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        panel_box = SurroundingRectangle(panel_content, buff=0.14, color=GREY_B, stroke_width=1.1)
        panel_box.set_fill(BLACK, opacity=0.92)
        panel = VGroup(panel_box, panel_content).to_edge(RIGHT, buff=0.35).set_y(1.18)

        x_dot = Dot(x_axis.n2p(x) + UP * 0.08, radius=0.07, color=BLUE_D)
        x_label = Text("x=0.36", font_size=15, color=BLUE_D).next_to(x_dot, UP, buff=0.05)

        q_cont_dot = Dot(q_axis.n2p(q_cont), radius=0.055, color=BLUE_D)
        q_cont_label = Text("x/s=3.6", font_size=15, color=BLUE_D).next_to(q_cont_dot, UP, buff=0.05)

        q_snap_dot = Dot(q_axis.n2p(float(q)), radius=0.07, color=ORANGE)
        q_snap_label = Text("q=4", font_size=15, color=ORANGE).next_to(q_snap_dot, UP, buff=0.05)

        x_hat_dot = Dot(x_axis.n2p(x_hat) + DOWN * 0.08, radius=0.075, color=ORANGE)
        x_hat_label = Text("x_hat=0.40", font_size=15, color=ORANGE).next_to(x_hat_dot, DOWN, buff=0.05)

        to_code_arrow = Arrow(
            x_axis.n2p(x) + UP * 0.12,
            q_axis.n2p(q_cont) + DOWN * 0.12,
            buff=0.03,
            stroke_width=2.5,
            color=BLUE_D,
            max_tip_length_to_length_ratio=0.14,
        )
        snap_arrow = Arrow(
            q_axis.n2p(q_cont) + UP * 0.06,
            q_axis.n2p(float(q)) + UP * 0.06,
            buff=0.03,
            stroke_width=2.5,
            color=ORANGE,
            max_tip_length_to_length_ratio=0.2,
        )
        back_arrow = Arrow(
            q_axis.n2p(float(q)) + DOWN * 0.12,
            x_axis.n2p(x_hat) + UP * 0.12,
            buff=0.03,
            stroke_width=2.5,
            color=ORANGE,
            max_tip_length_to_length_ratio=0.14,
        )

        err_seg = Line(x_axis.n2p(x), x_axis.n2p(x_hat), color=YELLOW_D, stroke_width=5)
        err_label = Text(f"|x_hat - x| = {err:.2f}", font_size=17, color=YELLOW_D).next_to(err_seg, UP, buff=0.05)

        self.play(Write(title), FadeIn(subtitle), run_time=0.9)
        self.play(Create(q_axis), FadeIn(q_axis_label), Create(q_l), Create(q_r), run_time=0.9)
        self.play(Create(x_axis), FadeIn(x_axis_label), FadeIn(x_levels), Create(x_l), Create(x_r), run_time=1.0)
        self.play(FadeIn(legend), FadeIn(panel), run_time=0.8)

        self.play(FadeIn(x_dot), FadeIn(x_label), run_time=0.45)
        self.play(Create(to_code_arrow), FadeIn(q_cont_dot), FadeIn(q_cont_label), run_time=0.7)
        self.play(Create(snap_arrow), Transform(q_cont_dot, q_snap_dot), Transform(q_cont_label, q_snap_label), run_time=0.75)
        self.play(Create(back_arrow), FadeIn(x_hat_dot), FadeIn(x_hat_label), run_time=0.75)

        subtitle_error = Text("Quantization error is the gap between x and x_hat", font_size=20).next_to(title, DOWN, buff=0.12)
        self.play(Transform(subtitle, subtitle_error), Create(err_seg), FadeIn(err_label), run_time=0.85)
        self.wait(0.9)
