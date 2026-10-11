"""Deterministic weight-only fake quantization for inspectable experiments."""
from dataclasses import dataclass

import torch


def validate_matrix(weight: torch.Tensor, name: str = "weight") -> None:
    if weight.ndim != 2 or min(weight.shape) == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if weight.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")
    if not torch.isfinite(weight).all():
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class QuantizationMetadata:
    bits: int
    scale: torch.Tensor
    quant_min: int
    quant_max: int
    scheme: str = "per_tensor_symmetric"
    rounding: str = "nearest_ties_to_even"


@dataclass(frozen=True)
class QuantizedWeight:
    original: torch.Tensor
    quantized: torch.Tensor
    metadata: QuantizationMetadata


class WeightQuantizer:
    """Round weights to a bounded symmetric grid, keeping arithmetic in float.

    Uses [-qmax, qmax], qmax = 2**(bits-1)-1 (one signed code unused),
    absmax/qmax scale, and PyTorch's ties-to-even rounding. Biases and
    activations are never passed through this quantizer. Zero matrices use
    scale=1. The result owns detached copies of both weight matrices.
    """

    def __init__(self, bits: int = 8):
        if isinstance(bits, bool) or not isinstance(bits, int) or not 2 <= bits <= 16:
            raise ValueError("bits must be an integer between 2 and 16")
        self.bits = bits

    @torch.no_grad()
    def __call__(self, weight: torch.Tensor) -> QuantizedWeight:
        validate_matrix(weight)
        original = weight.detach().clone()
        qmax = 2 ** (self.bits - 1) - 1
        absmax = original.abs().max()
        scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
        if scale <= 0:
            raise ValueError("quantization scale underflowed; use float64 weights")
        codes = torch.round(original / scale).clamp(-qmax, qmax)
        return QuantizedWeight(
            original, codes * scale,
            QuantizationMetadata(self.bits, scale, -qmax, qmax),
        )
