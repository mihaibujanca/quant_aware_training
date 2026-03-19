import math

import numpy as np
import pytest
import torch

from aleph.boundary_geometric_compression import (
    BoundaryCompressionConfig,
    _extract_longest_contour,
    _subspace_alignment_score,
    build_hybrid_boundary_model,
    build_teacher_boundary_profile,
    extract_teacher_projection,
    run_single_boundary_compression,
)
from aleph.datasets import (
    make_embedded_boundary_classification_benchmark,
)
from aleph.models import MLP


class ManualBoundaryTeacher(torch.nn.Module):
    def __init__(self, embedding, shape: str):
        super().__init__()
        self.shape = shape
        W = np.asarray(embedding.W, dtype=np.float32)
        b = np.asarray(embedding.b, dtype=np.float32)
        self.register_buffer("W_pinv", torch.tensor(np.linalg.pinv(W), dtype=torch.float32))
        self.register_buffer("b", torch.tensor(b, dtype=torch.float32))

    def _radius(self, theta: torch.Tensor) -> torch.Tensor:
        if self.shape == "flower":
            return 1.0 + 0.18 * torch.cos(6.0 * theta)
        if self.shape == "rounded_star":
            return 0.90 + 0.22 * torch.cos(5.0 * theta) + 0.10 * torch.cos(10.0 * theta)
        if self.shape == "dumbbell":
            return 0.82 + 0.33 * torch.cos(2.0 * theta)
        if self.shape == "windowed_ripple_ring":
            base = 1.02 + 0.05 * torch.cos(2.0 * theta - 0.35)
            delta_a = torch.remainder(theta + 1.05 + math.pi, 2.0 * math.pi) - math.pi
            delta_b = torch.remainder(theta - 1.42 + math.pi, 2.0 * math.pi) - math.pi
            win_a = torch.where(
                delta_a.abs() <= 0.48,
                0.5 * (1.0 + torch.cos(math.pi * delta_a / 0.48)),
                torch.zeros_like(theta),
            )
            win_b = torch.where(
                delta_b.abs() <= 0.42,
                0.5 * (1.0 + torch.cos(math.pi * delta_b / 0.42)),
                torch.zeros_like(theta),
            )
            return base + 0.12 * win_a * torch.cos(18.0 * (theta + 0.12)) + 0.10 * win_b * torch.cos(16.0 * (theta - 0.28))
        if self.shape == "corrugated_dumbbell":
            base = 0.84 + 0.31 * torch.cos(2.0 * theta + 0.08) + 0.04 * torch.cos(6.0 * theta - 0.15)
            delta = torch.remainder(theta - 0.62 + math.pi, 2.0 * math.pi) - math.pi
            win = torch.where(
                delta.abs() <= 0.46,
                0.5 * (1.0 + torch.cos(math.pi * delta / 0.46)),
                torch.zeros_like(theta),
            )
            return base + 0.12 * win * torch.cos(16.0 * (theta - 0.62))
        raise ValueError(self.shape)

    def forward(self, x_high: torch.Tensor) -> torch.Tensor:
        x_low = (x_high - self.b) @ self.W_pinv
        radius = torch.linalg.norm(x_low, dim=1)
        theta = torch.atan2(x_low[:, 1], x_low[:, 0])
        logits = self._radius(theta) - radius
        return logits.unsqueeze(1)


def _train_smoke_teacher(benchmark, epochs: int = 180) -> MLP:
    torch.manual_seed(0)
    model = MLP(input_size=benchmark.observed_dim, hidden_size=48, output_size=1, depth=3)
    X = torch.tensor(benchmark.train.X_high, dtype=torch.float32)
    y = torch.tensor(benchmark.train.y, dtype=torch.float32).unsqueeze(1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()
    return model


class TestBoundaryBenchmarkContract:
    def test_boundary_benchmark_exposes_boundary_and_curvature(self):
        for dataset in ("windowed_ripple_ring", "serpentine_tube", "multi_island"):
            benchmark = make_embedded_boundary_classification_benchmark(
                dataset,
                observed_dim=100,
                n_samples=240,
                random_state=7,
            )
            assert benchmark.boundary_points_low is not None
            assert benchmark.true_curvature_low is not None
            assert benchmark.boundary_points_low.shape[1] == 2
            assert benchmark.true_curvature_low.shape[0] == benchmark.boundary_points_low.shape[0]

            recovered = benchmark.embedding.inverse_transform(benchmark.train.X_high[:32])
            np.testing.assert_allclose(recovered, benchmark.train.X_low[:32], atol=1e-4, rtol=1e-4)

    def test_harder_benchmarks_have_localized_curvature_peaks(self):
        for dataset in ("windowed_ripple_ring", "serpentine_tube", "corrugated_dumbbell"):
            benchmark = make_embedded_boundary_classification_benchmark(
                dataset,
                observed_dim=2,
                n_samples=240,
                random_state=9,
            )
            curv = benchmark.true_curvature_low
            q50 = float(np.median(curv))
            q95 = float(np.quantile(curv, 0.95))
            assert q95 > max(1.0, 3.0 * q50)


class TestProjectionRecovery:
    def test_projection_methods_recover_embedding_subspace(self):
        benchmark = make_embedded_boundary_classification_benchmark(
            "rounded_star",
            observed_dim=100,
            n_samples=256,
            random_state=11,
        )
        teacher = ManualBoundaryTeacher(benchmark.embedding, benchmark.name)
        for projection_method, min_score in (("manifold_pca", 0.95), ("jacobian_svd", 0.90)):
            projection_P, _ = extract_teacher_projection(
                teacher,
                benchmark.train.X_high,
                projection_rank=2,
                projection_method=projection_method,
                device=torch.device("cpu"),
            )
            score = _subspace_alignment_score(np.asarray(benchmark.embedding.W, dtype=np.float32), projection_P)
            assert score > min_score


class TestBoundaryExtraction:
    def test_marching_squares_reconstructs_known_boundary(self):
        benchmark = make_embedded_boundary_classification_benchmark(
            "flower",
            observed_dim=2,
            n_samples=256,
            random_state=5,
        )
        grid_x = np.linspace(-1.6, 1.6, 160, dtype=np.float32)
        grid_y = np.linspace(-1.6, 1.6, 160, dtype=np.float32)
        xx, yy = np.meshgrid(grid_x, grid_y)
        theta = np.arctan2(yy, xx)
        radius = np.sqrt(xx ** 2 + yy ** 2)
        boundary_radius = 1.0 + 0.18 * np.cos(6.0 * theta)
        logits = (boundary_radius - radius).astype(np.float32)
        contour = _extract_longest_contour(logits, grid_x, grid_y)
        assert contour.shape[1] == 2

        from sklearn.neighbors import NearestNeighbors

        nn_true = NearestNeighbors(n_neighbors=1).fit(benchmark.boundary_points_low)
        dist = nn_true.kneighbors(contour[:-1], return_distance=True)[0]
        assert float(dist.mean()) < 0.08


class TestHybridBoundaryCompression:
    def test_hybrid_model_respects_budget_and_allocates_splines_to_hard_chunks(self):
        benchmark = make_embedded_boundary_classification_benchmark(
            "flower",
            observed_dim=2,
            n_samples=320,
            random_state=17,
        )
        teacher = _train_smoke_teacher(benchmark, epochs=220)
        cfg = BoundaryCompressionConfig(
            observed_dims=(2,),
            n_samples=320,
            source_epochs=220,
            baseline_epochs=220,
            grid_size=128,
            target_trainable_cap=160,
            fallback_trainable_cap=224,
            seed=17,
        )
        profile = build_teacher_boundary_profile(teacher, benchmark, cfg, device=torch.device("cpu"))
        hybrid = build_hybrid_boundary_model(profile, benchmark, cfg, target_budget=160)

        assert hybrid.trainable_params <= 160
        assert hybrid.stored_numbers >= hybrid.trainable_params
        assert all(chunk.kind == "spline" for chunk in hybrid.high_curvature_chunks)
        if hybrid.high_curvature_chunks:
            mean_high = np.mean([chunk.mean_complexity for chunk in hybrid.high_curvature_chunks])
            mean_low = np.mean([chunk.mean_complexity for chunk in hybrid.low_curvature_chunks]) if hybrid.low_curvature_chunks else 0.0
            assert mean_high >= mean_low


class TestBoundaryCompressionSmoke:
    def test_single_run_smoke(self):
        cfg = BoundaryCompressionConfig(
            datasets=("windowed_ripple_ring",),
            observed_dims=(20,),
            gate_observed_dim=20,
            n_samples=220,
            source_hidden=48,
            source_depth=3,
            source_epochs=120,
            small_hidden=24,
            small_depth=3,
            baseline_epochs=120,
            grid_size=96,
            target_trainable_cap=160,
            fallback_trainable_cap=224,
            calibration_steps=80,
            seed=3,
        )
        result = run_single_boundary_compression(cfg, dataset="windowed_ripple_ring", observed_dim=20)

        assert result["benchmark"]["observed_dim"] == 20
        assert "hybrid_boundary" in result["metrics"]
        assert result["metrics"]["hybrid_boundary"]["trainable_params"] <= 224
        assert "true_curvature" in result["metrics"]["hybrid_boundary"]["bucket_metrics"]
        assert math.isfinite(result["teacher_profile"]["complexity_correlation"])

    def test_multi_island_is_rejected_by_single_contour_compressor(self):
        cfg = BoundaryCompressionConfig(
            datasets=("multi_island",),
            observed_dims=(20,),
            gate_observed_dim=20,
            n_samples=220,
            source_epochs=40,
            baseline_epochs=40,
            seed=5,
        )
        with pytest.raises(ValueError, match="single connected contour"):
            run_single_boundary_compression(cfg, dataset="multi_island", observed_dim=20)
