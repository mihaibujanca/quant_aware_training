from __future__ import annotations

import numpy as np
from manim import (
    BLUE_D,
    GREEN_D,
    GREY_B,
    MathTex,
    ORANGE,
    RED_D,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    FadeIn,
    FadeOut,
    Create,
    Scene,
    Text,
    Transform,
    Write,
    VGroup,
    Dot,
)

from .common import (
    NARR_BS,
    NARR_WS,
    NARR_WS_Q,
    apply_affine_transport,
    fit_affine_transport,
    narr_fitted_axes,
    relu,
    sample_circle,
)


class LinearCorrectionScene(Scene):
    """3) Linear correction is exact in pre-activation space, then becomes piecewise after ReLU."""

    def construct(self) -> None:
        title = Text("Linear Correction Lives in Pre-Activation Space", font_size=31).to_edge(UP)
        subtitle = Text("Phase 1: pre-activation z (single affine map is exact)", font_size=19)
        subtitle.next_to(title, DOWN, buff=0.18)
        method_title = Text("Fit by L2 least squares (full precision)", font_size=16, color=GREY_B)
        objective_pre = MathTex(
            r"\min_{M,t}\ \sum_i \|M z_i^{(q)} + t - z_i^{(f)}\|_2^2",
            font_size=22,
        )
        method_group = VGroup(method_title, objective_pre).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        method_group.to_edge(RIGHT, buff=0.25)
        method_group.set_y(subtitle.get_bottom()[1] - 1.0)
        objective_post = MathTex(
            r"\min_{M,t}\ \sum_i \|M a_i^{(q)} + t - a_i^{(f)}\|_2^2",
            font_size=22,
        )
        objective_post.move_to(objective_pre)

        layer_idx = 0
        points = sample_circle(1.5, 220)
        W, Wq, b = NARR_WS[layer_idx], NARR_WS_Q[layer_idx], NARR_BS[layer_idx]

        z_f = points @ W.T + b
        z_q = points @ Wq.T + b
        a_f = relu(z_f)
        a_q = relu(z_q)

        pre_linear, pre_offset = fit_affine_transport(z_q, z_f)
        z_corr = apply_affine_transport(z_q, pre_linear, pre_offset)

        post_linear, post_offset = fit_affine_transport(a_q, a_f)
        a_corr = apply_affine_transport(a_q, post_linear, post_offset)

        sign_match = np.all((z_f > 0.0) == (z_q > 0.0), axis=1)
        sign_cross = ~sign_match

        ax = narr_fitted_axes([z_f, z_q, a_f, a_q, z_corr, a_corr], size=6.5, pad_frac=0.2).shift(DOWN * 0.1)
        dot_radius = 0.031

        def dots_from(points_arr: np.ndarray, color, fill_opacity: float = 0.82) -> VGroup:
            return VGroup(
                *[
                    Dot(
                        ax.c2p(float(points_arr[i, 0]), float(points_arr[i, 1])),
                        radius=dot_radius,
                        color=color,
                        fill_opacity=fill_opacity,
                    )
                    for i in range(len(points_arr))
                ]
            )

        pre_float_dots = dots_from(z_f, BLUE_D, fill_opacity=0.45)
        pre_quant_dots = dots_from(z_q, ORANGE)
        pre_corr_dots = dots_from(z_corr, GREEN_D)

        post_float_dots = dots_from(a_f, BLUE_D, fill_opacity=0.45)
        post_quant_dots = dots_from(a_q, ORANGE)
        post_corr_dots = VGroup(
            *[
                Dot(
                    ax.c2p(float(a_corr[i, 0]), float(a_corr[i, 1])),
                    radius=dot_radius,
                    color=GREEN_D if sign_match[i] else RED_D,
                    fill_opacity=0.86,
                )
                for i in range(len(a_corr))
            ]
        )

        legend_pre = VGroup(
            Text("blue: float z", font_size=16, color=BLUE_D),
            Text("orange: quantized z", font_size=16, color=ORANGE),
            Text("green: affine-corrected z", font_size=16, color=GREEN_D),
        ).arrange(RIGHT, buff=0.42).to_edge(DOWN, buff=0.72)

        legend_post = VGroup(
            Text("blue: float a=ReLU(z)", font_size=16, color=BLUE_D),
            Text("green: same ReLU region", font_size=16, color=GREEN_D),
            Text("red: crossed ReLU region", font_size=16, color=RED_D),
        ).arrange(RIGHT, buff=0.42).to_edge(DOWN, buff=0.72)

        pre_before = float(np.mean(np.linalg.norm(z_q - z_f, axis=1)))
        pre_after = float(np.mean(np.linalg.norm(z_corr - z_f, axis=1)))
        pre_max = float(np.max(np.linalg.norm(z_corr - z_f, axis=1)))

        pre_stats_before = Text(
            f"pre-activation: mean L2 before = {pre_before:.3f}",
            font_size=16,
            color=ORANGE,
        ).to_edge(DOWN, buff=0.32)
        pre_stats_after = Text(
            f"pre-activation after affine fit: mean = {pre_after:.2e}, max = {pre_max:.2e}",
            font_size=16,
            color=GREEN_D,
        ).to_edge(DOWN, buff=0.32)

        post_before = float(np.mean(np.linalg.norm(a_q - a_f, axis=1)))
        post_after = float(np.mean(np.linalg.norm(a_corr - a_f, axis=1)))
        if np.any(sign_match):
            post_same = float(np.mean(np.linalg.norm(a_corr[sign_match] - a_f[sign_match], axis=1)))
        else:
            post_same = 0.0
        if np.any(sign_cross):
            post_cross = float(np.mean(np.linalg.norm(a_corr[sign_cross] - a_f[sign_cross], axis=1)))
            cross_n = int(np.sum(sign_cross))
        else:
            post_cross = 0.0
            cross_n = 0

        post_stats_before = Text(
            f"post-ReLU: mean L2 before = {post_before:.3f}",
            font_size=16,
            color=ORANGE,
        ).to_edge(DOWN, buff=0.32)
        post_stats_after = Text(
            f"single affine on post-ReLU: mean = {post_after:.3f}, same/cross = {post_same:.3f}/{post_cross:.3f} ({cross_n} crossed)",
            font_size=16,
            color=RED_D,
        ).to_edge(DOWN, buff=0.32)

        self.play(Write(title), FadeIn(subtitle), FadeIn(method_title), FadeIn(objective_pre), run_time=0.9)
        self.play(Create(ax), FadeIn(legend_pre[:2]), run_time=0.9)
        self.play(FadeIn(pre_float_dots, lag_ratio=0.005), run_time=0.9)
        self.play(FadeIn(pre_quant_dots, lag_ratio=0.005), FadeIn(pre_stats_before), run_time=1.0)
        self.play(FadeOut(pre_stats_before), run_time=0.2)
        self.play(Transform(pre_quant_dots, pre_corr_dots), FadeIn(legend_pre[2]), run_time=1.25)
        self.play(FadeIn(pre_stats_after), run_time=0.35)
        self.wait(0.65)

        subtitle_post = Text("Phase 2: after ReLU, relation is piecewise-affine (single affine is imperfect)", font_size=19)
        subtitle_post.next_to(title, DOWN, buff=0.18)
        self.play(
            Transform(subtitle, subtitle_post),
            Transform(objective_pre, objective_post),
            FadeOut(pre_stats_after),
            FadeOut(legend_pre),
            run_time=0.65,
        )
        self.play(
            Transform(pre_float_dots, post_float_dots),
            Transform(pre_quant_dots, post_quant_dots),
            FadeIn(legend_post[0]),
            FadeIn(post_stats_before),
            run_time=1.25,
        )
        self.play(FadeOut(post_stats_before), run_time=0.2)
        self.play(Transform(pre_quant_dots, post_corr_dots), FadeIn(legend_post[1:]), run_time=1.25)
        self.play(FadeIn(post_stats_after), run_time=0.4)
        self.wait(1.2)
