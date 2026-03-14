import unittest

import numpy as np

from aleph.geometric_compression import (
    GeometricCompressionConfig,
    _kan_k_to_rbf_width_scale,
    run_pipeline,
)


class GeometricCompressionSpeedTests(unittest.TestCase):
    def _tiny_cfg(self, **kwargs) -> GeometricCompressionConfig:
        base = dict(
            mesh_name="sphere",
            n_points=1200,
            hidden_dim=64,
            depth=2,
            epochs=2,
            kan_fit_steps=2,
            n_profile=200,
            n_fit=160,
            grid_res=16,
            batch_size=512,
            device="cpu",
            seed=0,
            n_perturbations=2,
            kmeans_n_init=2,
            lbfgs_max_iter=5,
            eikonal_every_n_steps=3,
            eikonal_subsample=64,
            speed_tier=None,
        )
        base.update(kwargs)
        return GeometricCompressionConfig(**base)

    def test_kan_k_mapping_clamps_and_scales(self):
        self.assertAlmostEqual(_kan_k_to_rbf_width_scale(3), 1.0)
        self.assertAlmostEqual(_kan_k_to_rbf_width_scale(1), 1.0 / 3.0)
        self.assertAlmostEqual(_kan_k_to_rbf_width_scale(6), 2.0)
        self.assertAlmostEqual(_kan_k_to_rbf_width_scale(100), 4.0)
        self.assertAlmostEqual(_kan_k_to_rbf_width_scale(0), 0.25)

    def test_methods_gating_runs_only_requested_models(self):
        cfg = self._tiny_cfg(
            methods=("source_mlp", "geometric_kan"),
            compute_surface_metrics=False,
        )
        result = run_pipeline(cfg)
        self.assertEqual(set(result["metrics"].keys()), {"source_mlp", "geometric_kan"})
        for vals in result["metrics"].values():
            self.assertNotIn("chamfer", vals)
            self.assertNotIn("hausdorff", vals)

    def test_surface_metric_methods_limit_which_models_get_mesh_metrics(self):
        cfg = self._tiny_cfg(
            methods=("source_mlp", "geometric_kan", "small_mlp"),
            compute_surface_metrics=True,
            surface_metric_methods=("geometric_kan",),
        )
        result = run_pipeline(cfg)
        self.assertIn("chamfer", result["metrics"]["geometric_kan"])
        self.assertNotIn("chamfer", result["metrics"]["source_mlp"])
        self.assertNotIn("chamfer", result["metrics"]["small_mlp"])

    def test_eikonal_throttling_outputs_finite_metrics(self):
        cfg = self._tiny_cfg(
            methods=("source_mlp", "geometric_kan"),
            compute_surface_metrics=False,
            eikonal_every_n_steps=5,
            eikonal_subsample=32,
        )
        result = run_pipeline(cfg)
        self.assertTrue(np.isfinite(result["geometric_profile"]["source_test_mse"]))
        self.assertTrue(np.isfinite(result["metrics"]["geometric_kan"]["test_mse"]))
        self.assertTrue(np.isfinite(result["metrics"]["geometric_kan"]["jacobian_mse"]))


if __name__ == "__main__":
    unittest.main()
