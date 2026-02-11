from __future__ import annotations

import numpy as np
from manim import BLUE_D, GREEN_D, RED_D, DOWN, RIGHT, UP, FadeIn, Create, Scene, Text, Write, VGroup, Dot, Arrow

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


class TopologyFailureScene(Scene):
    """4) Visualize where linear correction fails: topology changed."""

    def construct(self) -> None:
        title = Text("Where Linear Correction Fails: Topology Changed", font_size=30).to_edge(UP)

        points = sample_circle(1.5, 220)
        layer_idx = 0
        W, Wq, b = NARR_WS[layer_idx], NARR_WS_Q[layer_idx], NARR_BS[layer_idx]

        z_f = points @ W.T + b
        z_q = points @ Wq.T + b
        sign_match = np.all((z_f > 0.0) == (z_q > 0.0), axis=1)
        sign_cross = ~sign_match

        a_f = relu(z_f)
        a_q = relu(z_q)
        linear, offset = fit_affine_transport(a_q, a_f)
        a_corr = apply_affine_transport(a_q, linear, offset)

        ax = narr_fitted_axes([a_f, a_corr], size=6.5, pad_frac=0.18).shift(DOWN * 0.12)

        dot_radius = 0.032
        float_dots = VGroup(
            *[
                Dot(ax.c2p(float(a_f[i, 0]), float(a_f[i, 1])), radius=dot_radius, color=BLUE_D, fill_opacity=0.4)
                for i in range(len(points))
            ]
        )
        corr_dots = VGroup(
            *[
                Dot(
                    ax.c2p(float(a_corr[i, 0]), float(a_corr[i, 1])),
                    radius=dot_radius,
                    color=GREEN_D if sign_match[i] else RED_D,
                    fill_opacity=0.85,
                )
                for i in range(len(points))
            ]
        )

        topo_arrows = VGroup()
        for idx in np.where(sign_cross)[0]:
            yf, yc = a_f[idx], a_corr[idx]
            if np.linalg.norm(yc - yf) < 1e-6:
                continue
            topo_arrows.add(
                Arrow(
                    ax.c2p(float(yf[0]), float(yf[1])),
                    ax.c2p(float(yc[0]), float(yc[1])),
                    buff=0.0,
                    stroke_width=2.3,
                    max_tip_length_to_length_ratio=0.24,
                    color=RED_D,
                )
            )

        topo_n = int(np.sum(sign_cross))
        total = len(points)
        topo_pct = 100.0 * topo_n / total
        topo_res = float(np.mean(np.linalg.norm(a_corr[sign_cross] - a_f[sign_cross], axis=1))) if topo_n > 0 else 0.0
        metric_res = float(np.mean(np.linalg.norm(a_corr[sign_match] - a_f[sign_match], axis=1)))

        subtitle = Text(
            "Exact continuation of Scene 3 (same layer, same LSQ fit), now color by region agreement",
            font_size=18,
        ).next_to(title, DOWN, buff=0.18)

        legend = VGroup(
            Text("blue: float", font_size=16, color=BLUE_D),
            Text("green: same ReLU regions", font_size=16, color=GREEN_D),
            Text("red: crossed ReLU region", font_size=16, color=RED_D),
        ).arrange(RIGHT, buff=0.45)
        legend.to_edge(DOWN, buff=0.72)

        stats = Text(
            f"topological points: {topo_n}/{total} ({topo_pct:.1f}%),  residual metric/topological: {metric_res:.3f}/{topo_res:.3f}",
            font_size=16,
            color=RED_D,
        ).to_edge(DOWN, buff=0.32)

        self.play(Write(title), FadeIn(subtitle), run_time=0.9)
        self.play(Create(ax), FadeIn(legend), run_time=0.9)
        self.play(FadeIn(float_dots, lag_ratio=0.005), run_time=1.0)
        self.play(FadeIn(corr_dots, lag_ratio=0.005), run_time=1.0)
        self.play(Create(topo_arrows, lag_ratio=0.03), FadeIn(stats), run_time=1.1)
        self.wait(1.2)
