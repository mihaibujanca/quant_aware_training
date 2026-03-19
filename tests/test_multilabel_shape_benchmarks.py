import numpy as np

from aleph.datasets import make_embedded_multilabel_shape_benchmark


class TestMultiLabelShapeBenchmark:
    def test_benchmark_exposes_embedded_multilabel_contract(self):
        benchmark = make_embedded_multilabel_shape_benchmark(
            dataset="overlap_triplet",
            observed_dim=100,
            n_samples=320,
            random_state=7,
        )
        assert benchmark.intrinsic_dim == 2
        assert benchmark.observed_dim == 100
        assert benchmark.train.X_low.shape[1] == 2
        assert benchmark.train.X_high.shape[1] == 100
        assert benchmark.train.y.shape[1] == 3
        assert len(benchmark.label_names) == 3

        recovered = benchmark.embedding.inverse_transform(benchmark.train.X_high[:32])
        np.testing.assert_allclose(recovered, benchmark.train.X_low[:32], atol=1e-4, rtol=1e-4)

    def test_labels_are_nontrivial_and_overlapping(self):
        benchmark = make_embedded_multilabel_shape_benchmark(
            dataset="rough_overlap_triplet",
            observed_dim=2,
            n_samples=600,
            random_state=11,
        )
        Y = benchmark.train.y
        prevalence = Y.mean(axis=0)
        assert np.all(prevalence > 0.05)
        assert np.all(prevalence < 0.95)

        overlap_fraction = (Y.sum(axis=1) >= 2).mean()
        assert float(overlap_fraction) > 0.05

    def test_harder_pentad_exposes_five_labels(self):
        benchmark = make_embedded_multilabel_shape_benchmark(
            dataset="rough_overlap_pentad",
            observed_dim=2,
            n_samples=600,
            random_state=13,
        )
        assert benchmark.train.y.shape[1] == 5
        overlap_fraction = (benchmark.train.y.sum(axis=1) >= 2).mean()
        assert float(overlap_fraction) > 0.10
