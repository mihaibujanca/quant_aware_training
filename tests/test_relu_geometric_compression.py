import math

import numpy as np
import torch

from aleph.datasets import (
    EmbeddedClassificationBenchmark,
    EmbeddedClassificationSplit,
    IdentityEmbedding,
    make_embedded_classification_benchmark,
)
from aleph.models import MLP
from aleph.relu_geometric_compression import (
    HighDReluCompressionConfig,
    RegionAtlas,
    RegionInfo,
    build_region_atlas,
    coarsen_region_atlas,
    run_single_high_d_compression,
)


def _manual_mlp() -> MLP:
    model = MLP(input_size=2, hidden_size=4, output_size=1, depth=2)
    linears = [layer for layer in model.layers if isinstance(layer, torch.nn.Linear)]
    with torch.no_grad():
        linears[0].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.5],
                    [0.5, -1.0],
                ]
            )
        )
        linears[0].bias.copy_(torch.tensor([0.2, -0.1, 0.0, 0.1]))
        linears[1].weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.5, 0.0],
                    [0.5, -0.5, 1.0, 0.0],
                    [0.0, 0.5, 0.0, 1.0],
                ]
            )
        )
        linears[1].bias.copy_(torch.tensor([0.0, 0.1, -0.2, 0.0]))
        linears[2].weight.copy_(torch.tensor([[1.0, -0.5, 0.75, -0.25]]))
        linears[2].bias.copy_(torch.tensor([0.15]))
    return model


class TestEmbeddedBenchmarkContract:
    def test_benchmark_bundle_exposes_intrinsic_and_observed_views(self):
        for dataset in ("spirals", "rough_spirals"):
            benchmark = make_embedded_classification_benchmark(
                dataset=dataset,
                observed_dim=100,
                n_samples=200,
                random_state=7,
            )
            assert benchmark.intrinsic_dim == 2
            assert benchmark.observed_dim == 100
            assert benchmark.train.X_low.shape[1] == 2
            assert benchmark.train.X_high.shape[1] == 100
            assert benchmark.embedding.inverse_transform(benchmark.train.X_high[:32]).shape == (32, 2)

            recovered = benchmark.embedding.inverse_transform(benchmark.train.X_high[:32])
            np.testing.assert_allclose(recovered, benchmark.train.X_low[:32], atol=1e-4, rtol=1e-4)

    def test_higher_intrinsic_benchmarks_expose_expected_dimensions(self):
        expected_dims = {
            "rough_hypershell_3d": 3,
            "rough_hypershell_5d": 5,
        }
        for dataset, intrinsic_dim in expected_dims.items():
            benchmark = make_embedded_classification_benchmark(
                dataset=dataset,
                observed_dim=100,
                n_samples=320,
                random_state=9,
            )
            assert benchmark.intrinsic_dim == intrinsic_dim
            assert benchmark.observed_dim == 100
            assert benchmark.train.X_low.shape[1] == intrinsic_dim
            assert benchmark.train.X_high.shape[1] == 100

            recovered = benchmark.embedding.inverse_transform(benchmark.train.X_high[:32])
            np.testing.assert_allclose(recovered, benchmark.train.X_low[:32], atol=1e-4, rtol=1e-4)

            prevalence = float(benchmark.train.y.mean())
            assert 0.15 < prevalence < 0.85


class TestRegionAtlasExtraction:
    def test_affine_map_reproduces_outputs_on_region_support(self):
        benchmark = make_embedded_classification_benchmark(
            dataset="spirals",
            observed_dim=2,
            n_samples=160,
            random_state=11,
        )
        model = _manual_mlp()
        atlas = build_region_atlas(
            model,
            benchmark,
            device=torch.device("cpu"),
            knn_k=8,
            curvature_eps=0.03,
            boundary_eps=0.05,
            boundary_perturbations=4,
        )

        X_high = benchmark.train.X_high
        logits = atlas.sample_logits
        for region in atlas.regions:
            support = np.flatnonzero(atlas.sample_region_ids == region.region_id)
            if len(support) == 0:
                continue
            recon = X_high[support] @ region.affine_weights + region.affine_bias
            np.testing.assert_allclose(recon, logits[support], atol=1e-5, rtol=1e-5)

    def test_region_atlas_is_deterministic(self):
        benchmark = make_embedded_classification_benchmark(
            dataset="spirals",
            observed_dim=2,
            n_samples=160,
            random_state=11,
        )
        model = _manual_mlp()
        atlas_a = build_region_atlas(
            model,
            benchmark,
            device=torch.device("cpu"),
            knn_k=8,
            curvature_eps=0.03,
            boundary_eps=0.05,
            boundary_perturbations=4,
        )
        atlas_b = build_region_atlas(
            model,
            benchmark,
            device=torch.device("cpu"),
            knn_k=8,
            curvature_eps=0.03,
            boundary_eps=0.05,
            boundary_perturbations=4,
        )
        assert atlas_a.sample_activation_codes == atlas_b.sample_activation_codes
        np.testing.assert_array_equal(atlas_a.sample_region_ids, atlas_b.sample_region_ids)

    def test_region_adjacency_is_symmetric_and_connected_when_nontrivial(self):
        benchmark = make_embedded_classification_benchmark(
            dataset="spirals",
            observed_dim=2,
            n_samples=200,
            random_state=13,
        )
        model = _manual_mlp()
        atlas = build_region_atlas(
            model,
            benchmark,
            device=torch.device("cpu"),
            knn_k=10,
            curvature_eps=0.03,
            boundary_eps=0.05,
            boundary_perturbations=4,
        )

        for left, neighbors in atlas.adjacency.items():
            for right in neighbors:
                assert left in atlas.adjacency[right]

        if len(atlas.regions) > 1:
            seen = set()
            stack = [0]
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node)
                stack.extend(n for n in atlas.adjacency[node] if n not in seen)
            assert len(seen) == len(atlas.regions)


class TestMacroRegionCoarsening:
    def test_merge_policy_stays_within_same_predicted_class_and_low_complexity(self):
        benchmark = EmbeddedClassificationBenchmark(
            name="toy",
            intrinsic_dim=2,
            observed_dim=2,
            embedding=IdentityEmbedding(2),
            train=EmbeddedClassificationSplit(
                X_low=np.asarray([[0, 0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]], dtype=np.float32),
                X_high=np.asarray([[0, 0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]], dtype=np.float32),
                y=np.asarray([1, 1, 1, 0], dtype=np.int64),
            ),
            val=EmbeddedClassificationSplit(
                X_low=np.asarray([[0, 0], [0.1, 0.0]], dtype=np.float32),
                X_high=np.asarray([[0, 0], [0.1, 0.0]], dtype=np.float32),
                y=np.asarray([1, 1], dtype=np.int64),
            ),
            test=EmbeddedClassificationSplit(
                X_low=np.asarray([[0.2, 0.0], [0.3, 0.0]], dtype=np.float32),
                X_high=np.asarray([[0.2, 0.0], [0.3, 0.0]], dtype=np.float32),
                y=np.asarray([1, 0], dtype=np.int64),
            ),
        )
        atlas = RegionAtlas(
            benchmark_name="toy",
            observed_dim=2,
            sample_region_ids=np.asarray([0, 1, 2, 3], dtype=np.int64),
            sample_activation_codes=("a", "b", "c", "d"),
            sample_logits=np.asarray([2.0, 2.1, 2.2, -2.5], dtype=np.float32),
            sample_margins=np.asarray([2.0, 2.1, 2.2, 2.5], dtype=np.float32),
            sample_curvature=np.asarray([0.05, 0.07, 0.95, 0.90], dtype=np.float32),
            sample_boundary_density=np.asarray([0.10, 0.12, 1.20, 1.00], dtype=np.float32),
            sample_hardness=np.asarray([0.05, 0.07, 0.85, 0.99], dtype=np.float32),
            sample_activation_churn=np.asarray([0.0, 0.1, 0.8, 0.9], dtype=np.float32),
            sample_region_density=np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
            sample_jacobian_change=np.asarray([0.02, 0.03, 0.7, 0.8], dtype=np.float32),
            regions=(
                RegionInfo(0, "a", 1, 0.25, 1, 2.0, np.asarray([1.0, 0.0], dtype=np.float32), 0.0,
                           np.asarray([0.0, 0.0], dtype=np.float32), np.asarray([0.0, 0.0], dtype=np.float32),
                           0.05, 0.10, 0.05, 0.0, 1.0, 0.02),
                RegionInfo(1, "b", 1, 0.25, 1, 2.1, np.asarray([1.02, 0.0], dtype=np.float32), 0.0,
                           np.asarray([0.1, 0.0], dtype=np.float32), np.asarray([0.1, 0.0], dtype=np.float32),
                           0.07, 0.12, 0.07, 0.1, 1.0, 0.03),
                RegionInfo(2, "c", 1, 0.25, 1, 2.2, np.asarray([1.8, 0.4], dtype=np.float32), 0.0,
                           np.asarray([0.2, 0.0], dtype=np.float32), np.asarray([0.2, 0.0], dtype=np.float32),
                           0.95, 1.20, 0.85, 0.8, 2.0, 0.7),
                RegionInfo(3, "d", 1, 0.25, 0, 2.5, np.asarray([-1.0, 0.0], dtype=np.float32), 0.0,
                           np.asarray([0.3, 0.0], dtype=np.float32), np.asarray([0.3, 0.0], dtype=np.float32),
                           0.90, 1.00, 0.99, 0.9, 2.0, 0.8),
            ),
            adjacency={0: (1,), 1: (0, 2), 2: (1, 3), 3: (2,)},
            correlations={},
            stable_neuron_stats=(),
        )

        macro = coarsen_region_atlas(
            atlas,
            benchmark,
            max_affine_gap=0.10,
            max_merge_bce_increase=0.10,
            min_merge_margin=0.5,
            low_complexity_quantile=0.5,
        )

        member_sets = {tuple(sorted(region.member_regions)) for region in macro.macro_regions}
        assert (0, 1) in member_sets
        assert (2,) in member_sets
        assert (3,) in member_sets


class TestHighDReluCompressionSmoke:
    def test_single_run_returns_smaller_geometry_model(self):
        cfg = HighDReluCompressionConfig(
            observed_dims=(10,),
            n_samples=240,
            source_hidden=24,
            source_depth=3,
            source_epochs=60,
            baseline_epochs=20,
            cleanup_epochs=15,
            boundary_perturbations=4,
            knn_k=8,
            architecture_max_hidden=16,
            trace_every_n_steps=10,
            device="cpu",
            seed=3,
        )
        result = run_single_high_d_compression(cfg, observed_dim=10)
        metrics = result["metrics"]

        assert metrics["geometric_relu"]["params"] < metrics["source_mlp"]["params"]
        assert math.isfinite(metrics["geometric_relu"]["test_accuracy"])
        assert math.isfinite(metrics["distilled_relu"]["test_accuracy"])
        assert result["atlas"]["n_macro_regions"] <= result["atlas"]["n_regions"]
        assert len(result["training_traces"]["source_mlp"]) > 0
        assert result["training_traces"]["source_mlp"][-1]["step"] == float(cfg.source_epochs)
