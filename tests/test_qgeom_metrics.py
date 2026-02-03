import unittest

import numpy as np

from aleph.qgeom.metrics import (
    build_layer_geometry_report,
    compute_entropy_metrics,
    compute_fate_metrics,
    compute_geometry_metrics,
)


class MetricsTests(unittest.TestCase):
    def test_linear_only_case_has_tiny_nonlinear(self):
        linear = np.array([[0.1, -0.2], [0.05, 0.03]])
        nonlinear = np.zeros_like(linear)

        geom = compute_geometry_metrics(linear, nonlinear)
        self.assertAlmostEqual(geom["nonlinear_error_norm"], 0.0, places=8)

        fate = compute_fate_metrics(linear, linear)
        self.assertAlmostEqual(fate["relu_flip_rate"], 0.0, places=8)

        report = build_layer_geometry_report(
            0,
            linear_error=linear,
            nonlinear_error=nonlinear,
            float_pre_activation=linear,
            quant_pre_activation=linear,
        )
        self.assertGreater(report.correctability_score, 0.8)

    def test_forced_clipping_increases_nonlinear_path(self):
        float_pre = np.array([[0.9, 0.7], [0.8, 0.6]])
        quant_pre = np.array([[3.2, -3.1], [2.9, -2.7]])

        linear = np.zeros_like(float_pre)
        nonlinear = np.abs(quant_pre - float_pre)
        sat_mask = np.ones_like(float_pre, dtype=bool)

        report = build_layer_geometry_report(
            1,
            linear_error=linear,
            nonlinear_error=nonlinear,
            float_pre_activation=float_pre,
            quant_pre_activation=quant_pre,
            saturation_mask=sat_mask,
        )
        self.assertGreater(report.nonlinear_error_norm, report.linear_error_norm)
        self.assertGreater(report.saturation_count, 0)

    def test_relu_boundary_crossing_increases_flip_rate(self):
        float_pre = np.array([[0.01, -0.01], [0.02, -0.02], [0.01, 0.03]])
        quant_pre = -float_pre

        fate = compute_fate_metrics(float_pre, quant_pre)
        self.assertGreater(fate["relu_flip_rate"], 0.5)

    def test_entropy_metrics_finite_under_degenerate_cov(self):
        # Rank-1 signal in 4D.
        x = np.zeros((256, 4), dtype=float)
        x[:, 0] = np.linspace(-1, 1, num=256)

        ent = compute_entropy_metrics(x)
        self.assertTrue(np.isfinite(ent["entropy_proxy"]))
        self.assertTrue(np.isfinite(ent["volume_proxy"]))
        self.assertGreater(ent["effective_rank"], 0.0)


if __name__ == "__main__":
    unittest.main()
