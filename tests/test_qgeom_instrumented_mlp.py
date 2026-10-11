"""Causal and trace tests for the shared weight-only MLP runner."""
import pytest
import torch
from torch import nn

from aleph.models import MLP
from aleph.qgeom import CorrectionBridge, InstrumentedMLP, ResidualOracle, WeightQuantizer


def make_model(dtype=torch.float32):
    with torch.random.fork_rng():
        torch.manual_seed(24)
        return nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 2)).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_float_trace_matches_model_and_snapshot_is_detached(dtype):
    model = make_model(dtype)
    runner = InstrumentedMLP(model, quantizer=WeightQuantizer(2))
    x = torch.randn(9, 2, dtype=dtype)
    trace = runner.run(x, quantized_layers=[])
    torch.testing.assert_close(trace.output, model(x))
    assert len(trace.layers) == 3
    assert trace.layers[-1].relu_disagreement is None
    for layer in trace.layers:
        assert not layer.input.requires_grad
        torch.testing.assert_close(layer.ledger.total, torch.zeros_like(layer.ledger.total))
    original = trace.output.clone()
    with torch.no_grad():
        model[0].weight.fill_(9)
    torch.testing.assert_close(runner.run(x, quantized_layers=[]).output, original)


@pytest.mark.parametrize("source", [0, 1, 2])
def test_arbitrary_single_layer_quantization_and_immediate_oracle(source):
    runner = InstrumentedMLP(make_model(), quantizer=WeightQuantizer(2))
    x = torch.randn(23, 2, generator=torch.Generator().manual_seed(3))
    ordinary = runner.run(x, quantized_layers=[source])
    corrected = runner.run(x, quantized_layers=[source], bridges=[CorrectionBridge(source, source)])
    assert ordinary.layers[source].total_error.norm() > 0
    torch.testing.assert_close(corrected.output, corrected.fp.post_acts[-1], atol=1e-6, rtol=1e-5)
    for layer in corrected.layers[source + 1:]:
        torch.testing.assert_close(layer.quant_post, layer.fp_post, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(corrected.layers[source].quant_post, ordinary.layers[source].quant_post)
    torch.testing.assert_close(corrected.layers[source].oracle_residual, -ordinary.layers[source].total_error)
    for layer in ordinary.layers:
        assert layer.quantized == (layer.index == source)
        torch.testing.assert_close(layer.ledger.total + layer.propagated_error, layer.total_error,
                                   atol=1e-6, rtol=1e-5)


def test_all_layer_immediate_preserves_local_damage_and_resets_inputs():
    runner = InstrumentedMLP(make_model(), quantizer=WeightQuantizer(2))
    x = torch.randn(19, 2, generator=torch.Generator().manual_seed(3))
    trace = runner.run(x, bridges=[CorrectionBridge(i, i) for i in range(3)])
    for layer in trace.layers:
        torch.testing.assert_close(layer.input, layer.fp_input, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(layer.corrected_post, layer.fp_post, atol=1e-6, rtol=1e-5)
        assert layer.ledger.total.norm() > 0
        assert layer.correction_applied
        assert layer.quantization.quantized.dtype == x.dtype
    torch.testing.assert_close(trace.output, trace.fp.post_acts[-1], atol=1e-6, rtol=1e-5)


def test_delayed_site_only_restores_at_target():
    runner = InstrumentedMLP(make_model(), quantizer=WeightQuantizer(2))
    x = torch.randn(19, 2, generator=torch.Generator().manual_seed(3))
    trace = runner.run(x, quantized_layers=[0], bridges=[CorrectionBridge(0, 1)])
    assert trace.layers[0].total_error.norm() > 0
    assert not trace.layers[0].correction_applied
    assert trace.layers[1].total_error.norm() > 0
    torch.testing.assert_close(trace.output, trace.fp.post_acts[-1], atol=1e-6, rtol=1e-5)


def test_oracle_sign_biases_float_and_aleph_adapter():
    model = MLP(input_size=2, hidden_size=4, output_size=1, depth=2)
    runner = InstrumentedMLP(model, quantizer=WeightQuantizer(4))
    x = torch.ones(3, 2)
    trace = runner.run(x, quantized_layers=[])
    torch.testing.assert_close(trace.output, model(x))
    for b, layer in zip(runner.biases, (model.layers[0], model.layers[2], model.layers[4])):
        torch.testing.assert_close(b, layer.bias)
    oracle = ResidualOracle(trace.fp)
    perturbed = trace.fp.post_acts[0] + 2
    torch.testing.assert_close(oracle.residual(perturbed, 0), torch.full_like(perturbed, -2))
    torch.testing.assert_close(oracle.correct(perturbed, 0), trace.fp.post_acts[0])


def test_biasless_model_and_final_relu():
    model = nn.Sequential(nn.Linear(2, 3, bias=False), nn.ReLU())
    runner = InstrumentedMLP(model)
    trace = runner.run(torch.ones(2, 2), quantized_layers=[])
    torch.testing.assert_close(trace.output, model(torch.ones(2, 2)))
    assert trace.layers[0].relu_disagreement is not None


def test_invalid_architectures_and_interventions_rejected():
    for model in [nn.Sequential(), nn.Sequential(nn.Linear(2, 3), nn.Sigmoid()),
                  nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2)),
                  nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Dropout())]:
        with pytest.raises(ValueError):
            InstrumentedMLP(model)
    with pytest.raises(ValueError):
        CorrectionBridge(1, 0)
    runner = InstrumentedMLP(make_model())
    for kwargs in [dict(quantized_layers=[3]), dict(quantized_layers=[False]),
                   dict(quantized_layers=[], bridges=[CorrectionBridge(0, 1)]),
                   dict(bridges=[CorrectionBridge(0, 3)]),
                   dict(bridges=[CorrectionBridge(0, 1), CorrectionBridge(1, 1)])]:
        with pytest.raises(ValueError):
            runner.run(torch.ones(2, 2), **kwargs)
