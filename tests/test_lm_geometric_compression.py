import math

from aleph.lm_geometric_compression import (
    CompressionBudgetSpec,
    DistillationSpec,
    LearnedCompressorSpec,
    ProbeCorpusSpec,
    TeacherLoadSpec,
    build_probe_corpus,
    load_nanochat_teacher,
    probe_ffn_geometry,
    result_to_dict,
    run_ffn_distillation_baseline,
    run_ffn_geometric_compression,
    run_ffn_learned_compressor,
)


def _make_teacher():
    return load_nanochat_teacher(
        TeacherLoadSpec(
            source="hf_random",
            device="cpu",
            hf_config_overrides={
                "vocab_size": 128,
                "n_positions": 24,
                "n_ctx": 24,
                "n_embd": 16,
                "n_layer": 2,
                "n_head": 2,
                "n_inner": 32,
            },
        )
    )


def _make_corpus(teacher):
    docs = (
        "The quick brown fox jumps over the lazy dog. " * 4,
        "Geometry matters when compressing language models. " * 4,
        "Residual streams and FFN blocks can be probed analytically. " * 4,
        "Distillation is the baseline, not the main method. " * 4,
        "Compression should preserve useful local linear structure. " * 4,
    )
    return build_probe_corpus(
        ProbeCorpusSpec(
            teacher=teacher,
            text_chunks=docs,
            sequence_length=12,
            probe_train_sequences=8,
            probe_holdout_sequences=4,
            eval_val_sequences=4,
            dataset_name="unit-test-corpus",
        )
    )


class TestTeacherAndCorpus:
    def test_random_teacher_and_probe_corpus_build(self):
        teacher = _make_teacher()
        corpus = _make_corpus(teacher)

        assert teacher.model_family == "gpt2"
        assert teacher.layer_ids == (0, 1)
        assert corpus.probe_train.shape == (8, 13)
        assert corpus.eval_val.shape == (4, 13)


class TestGeometryProbe:
    def test_probe_ffn_geometry_returns_per_layer_stats(self):
        teacher = _make_teacher()
        corpus = _make_corpus(teacher)
        atlas = probe_ffn_geometry(teacher, corpus, max_tokens_per_layer=32, batch_size=2)

        assert len(atlas.layer_geometries) == 2
        for layer in atlas.layer_geometries:
            assert layer.input_sample.shape[0] <= 32
            assert layer.hidden_dim == 32
            assert math.isfinite(layer.channel_scores.mean().item())
            assert layer.jacobian_spectrum.ndim == 1


class TestCompressionPipelines:
    def test_geometric_pipeline_compresses_ffns(self):
        teacher = _make_teacher()
        corpus = _make_corpus(teacher)
        budget = CompressionBudgetSpec(hidden_fraction=0.5, min_hidden=8, max_hidden=16)
        result = run_ffn_geometric_compression(teacher, corpus, budget, batch_size=2, max_tokens_per_layer=32)

        assert result.method == "geometric"
        assert result.metrics["ffn_param_count"] < result.metrics["source_ffn_param_count"]
        assert math.isfinite(result.metrics["perplexity"])
        assert len(result.layer_metrics) == 2

    def test_distillation_baseline_runs(self):
        teacher = _make_teacher()
        corpus = _make_corpus(teacher)
        budget = CompressionBudgetSpec(hidden_fraction=0.5, min_hidden=8, max_hidden=16)
        result = run_ffn_distillation_baseline(
            DistillationSpec(
                teacher=teacher,
                corpus=corpus,
                budget=budget,
                batch_size=2,
                steps=4,
                lr=1e-2,
            )
        )

        assert result.method == "distilled"
        assert result.metrics["ffn_param_count"] < result.metrics["source_ffn_param_count"]
        assert math.isfinite(result.metrics["teacher_student_kl"])

    def test_learned_autoencoder_compressor_runs(self):
        teacher = _make_teacher()
        corpus = _make_corpus(teacher)
        budget = CompressionBudgetSpec(hidden_fraction=0.5, min_hidden=8, max_hidden=16)
        result = run_ffn_learned_compressor(
            LearnedCompressorSpec(
                teacher=teacher,
                corpus=corpus,
                budget=budget,
                code_dim=8,
                autoencoder_hidden=32,
                epochs=2,
                batch_size=32,
                lr=1e-2,
            )
        )

        assert result.method == "learned_autoencoder"
        assert result.metrics["ffn_param_count"] < result.metrics["source_ffn_param_count"]
        assert math.isfinite(result.metrics["perplexity"])

    def test_result_serialization_smoke(self):
        teacher = _make_teacher()
        corpus = _make_corpus(teacher)
        budget = CompressionBudgetSpec(hidden_fraction=0.5, min_hidden=8, max_hidden=16)
        result = run_ffn_geometric_compression(teacher, corpus, budget, batch_size=2, max_tokens_per_layer=32)
        payload = result_to_dict(result)

        assert payload["method"] == "geometric"
        assert "metrics" in payload
