"""Acceptance tests for the exact weight-only single-layer apparatus."""
import pytest
import torch

from aleph.qgeom import ErrorLedger, MapDecomposition, WeightQuantizer


@pytest.mark.parametrize("bits", [2, 4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quantizer_bounded_deterministic_and_owns_weights(bits, dtype):
    W = torch.tensor([[0., -1., .23], [.72, .51, -.15]], dtype=dtype)
    quantizer = WeightQuantizer(bits)
    result = quantizer(W)
    torch.testing.assert_close(result.quantized, quantizer(W).quantized, rtol=0, atol=0)
    codes = result.quantized / result.metadata.scale
    assert codes.min() >= result.metadata.quant_min
    assert codes.max() <= result.metadata.quant_max
    torch.testing.assert_close(codes, codes.round())
    assert (result.quantized - W).abs().max() <= result.metadata.scale / 2 + torch.finfo(dtype).eps
    W.fill_(9)
    assert result.original.max() < 9
    assert not result.quantized.requires_grad


def test_zero_weights_and_rounding_ties():
    quantizer = WeightQuantizer(2)
    result = quantizer(torch.zeros(2, 3))
    assert result.metadata.scale == 1
    assert torch.count_nonzero(result.quantized) == 0
    W = torch.tensor([[1., .5, -.5]])
    torch.testing.assert_close(quantizer(W).quantized, torch.tensor([[1., 0., 0.]]))


@pytest.mark.parametrize("bits", [True, 1, 17, 4.5])
def test_quantizer_rejects_invalid_bits(bits):
    with pytest.raises(ValueError):
        WeightQuantizer(bits)


@pytest.mark.parametrize("shape", [(3, 3), (2, 5), (5, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rectangular_decomposition_and_biased_ledger(shape, dtype):
    generator = torch.Generator().manual_seed(31)
    W = torch.randn(shape, generator=generator, dtype=dtype)
    Wq = WeightQuantizer(4)(W).quantized
    decomp = MapDecomposition.from_weights(W, Wq)
    x = torch.randn(17, shape[0], generator=generator, dtype=dtype)
    bias = torch.randn(shape[1], generator=generator, dtype=dtype)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(Wq @ decomp.R + decomp.U, W, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(Wq.T @ decomp.U, torch.zeros_like(Wq.T @ decomp.U), atol=tolerance, rtol=0)
    ledger = ErrorLedger.compute(x, decomp, bias=bias)
    torch.testing.assert_close(ledger.total, torch.relu(x @ Wq + bias) - torch.relu(x @ W + bias))
    torch.testing.assert_close(ledger.uncomp + ledger.nonlinear + ledger.range, ledger.total,
                               atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(ledger.gram().sum(), ledger.total.square().sum(dim=1).mean(),
                               atol=tolerance, rtol=tolerance)
    assert ledger.total.shape == (17, shape[1])


def test_rank_deficiency_missing_input_and_nonlinear_cancellation():
    W = torch.tensor([[1., -.35], [.28, .22]], dtype=torch.float64)
    Wq = WeightQuantizer(2)(W).quantized
    decomp = MapDecomposition.from_weights(W, Wq)
    assert decomp.rank == 1
    torch.testing.assert_close(decomp.U[1], W[1])
    # Same quantized preactivation, different FP response: missing x_2 information.
    x = torch.tensor([[1., 0.], [1., 2.], [-1., 2.]], dtype=W.dtype)
    ledger = ErrorLedger.compute(x, decomp)
    torch.testing.assert_close(ledger.quant_pre[0], ledger.quant_pre[1])
    assert not torch.equal(ledger.fp_post[0], ledger.fp_post[1])
    assert ledger.nonlinear.norm() > 0
    assert ledger.range.norm() > 0
    assert ledger.gram()[0, 1] < 0  # ReLU removes the negative shear.


def test_identity_activation_and_bias_adjustment():
    W = torch.tensor([[1., -.35], [.28, .22]], dtype=torch.float64)
    decomp = MapDecomposition.from_weights(W, WeightQuantizer(2)(W).quantized)
    x = torch.tensor([[1., 2.], [-1., -2.]], dtype=W.dtype)
    unbiased = ErrorLedger.compute(x, decomp, activation=lambda z: z)
    torch.testing.assert_close(unbiased.nonlinear, torch.zeros_like(unbiased.nonlinear))
    bias = torch.tensor([.2, -.1], dtype=W.dtype)
    biased = ErrorLedger.compute(x, decomp, bias=bias, activation=lambda z: z)
    # Affine transport belongs to the recoverable correction; an identity
    # activation must not create artificial "nonlinear" error from bias.
    torch.testing.assert_close(biased.nonlinear, torch.zeros_like(biased.nonlinear))
    torch.testing.assert_close(biased.uncomp, unbiased.uncomp)


@pytest.mark.parametrize("shape", [(2, 4), (4, 2)])
def test_no_quantization_error_is_zero(shape):
    W = torch.randn(shape, generator=torch.Generator().manual_seed(42))
    decomp = MapDecomposition.from_weights(W, W)
    ledger = ErrorLedger.compute(torch.ones(7, shape[0]), decomp, bias=torch.ones(shape[1]))
    for component in ledger.components().values():
        torch.testing.assert_close(component, torch.zeros_like(component))
    assert decomp.diagnostics()["U_rank"] == 0


def test_zero_quantized_map_and_explicit_rank_tolerance():
    W = torch.eye(2, dtype=torch.float64)
    decomp = MapDecomposition.from_weights(W, torch.zeros_like(W))
    assert decomp.rank == 0
    torch.testing.assert_close(decomp.U, W)
    ErrorLedger.compute(torch.tensor([[1., -1.]], dtype=W.dtype), decomp)
    near_null = torch.diag(torch.tensor([1., 1e-8], dtype=W.dtype))
    truncated = MapDecomposition.from_weights(W, near_null, rtol=1e-6)
    retained = MapDecomposition.from_weights(W, near_null, rtol=1e-10)
    assert truncated.rank == 1
    assert retained.rank == 2
    torch.testing.assert_close(truncated.quantized_weight @ truncated.R + truncated.U, W)


def test_telescoping_fails_loudly_for_invalid_activation():
    W = torch.eye(2)
    decomp = MapDecomposition.from_weights(W, W)
    with pytest.raises(AssertionError):
        ErrorLedger.compute(torch.ones(1, 2), decomp, activation=lambda z: z * float("nan"))


def test_input_validation():
    with pytest.raises(ValueError):
        WeightQuantizer()(torch.tensor([[float("nan")]]))
    W = torch.eye(2)
    with pytest.raises(ValueError):
        MapDecomposition.from_weights(W, torch.ones(2, 3))
    decomp = MapDecomposition.from_weights(W, W)
    with pytest.raises(ValueError):
        ErrorLedger.compute(torch.ones(2, 3), decomp)
    with pytest.raises(ValueError):
        ErrorLedger.compute(torch.ones(2, 2), decomp, bias=torch.ones(3))
