"""Shared weight-only ReLU MLP traces and exact residual interventions.

Snapshot a float nn.Sequential or aleph.models.MLP once; select quantized
layers and correction sites independently on each run. Linear weights are
transposed at this boundary to the geometry modules' [input, output] format.
"""
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from aleph.qgeom.canonical import ForwardTrace
from aleph.qgeom.ledger import ErrorLedger
from aleph.qgeom.map_decomposition import MapDecomposition
from aleph.qgeom.weight_quantizer import QuantizedWeight, WeightQuantizer


@dataclass(frozen=True)
class CorrectionBridge:
    """An exact post-activation correction edge, using zero-based linear indices.

    target=source restores that layer's output before the next linear layer
    consumes it. target>source restores at a later layer. The exact residual
    restores the entire FP state at target, including ALL accumulated sources;
    it is source-specific only if source is the sole perturbed layer.
    """
    source: int
    target: int

    def __post_init__(self):
        if any(isinstance(i, bool) or not isinstance(i, int) for i in (self.source, self.target)):
            raise ValueError("bridge indices must be integers")
        if self.source < 0 or self.target < self.source:
            raise ValueError("bridge requires 0 <= source <= target")


class ResidualOracle:
    """Exact additive FP-minus-Q residual; requires a reference FP trace."""

    def __init__(self, reference: ForwardTrace):
        self.reference = reference

    def residual(self, h_q: torch.Tensor, layer: int) -> torch.Tensor:
        return self.reference.post_acts[layer] - h_q

    def correct(self, h_q: torch.Tensor, layer: int) -> torch.Tensor:
        return h_q + self.residual(h_q, layer)


@dataclass(frozen=True)
class LayerTrace:
    index: int
    name: str
    activation: str
    quantized: bool
    quantization: QuantizedWeight | None
    input: torch.Tensor
    fp_input: torch.Tensor
    fp_pre: torch.Tensor
    fp_post: torch.Tensor
    quant_pre: torch.Tensor  # raw, BEFORE intervention
    quant_post: torch.Tensor  # raw, BEFORE intervention
    corrected_post: torch.Tensor  # consumed by next layer
    oracle_residual: torch.Tensor  # available even when not applied
    correction_applied: bool
    ledger: ErrorLedger  # local creation on actual run input
    propagated_error: torch.Tensor  # same-input FP output minus reference FP
    total_error: torch.Tensor  # raw output minus reference FP
    relu_disagreement: torch.Tensor | None


@dataclass(frozen=True)
class ActivationTrace:
    """Aligned reference, raw and consumed states for an intervention run.

    raw_quantized uses the run's inputs, which may have been corrected at
    earlier sites. It is an ordinary Q trajectory only with no bridges.
    Correction is post-activation: raw preactivations remain inspectable.
    """
    fp: ForwardTrace
    raw_quantized: ForwardTrace
    consumed_post_acts: list[torch.Tensor]
    layers: list[LayerTrace]
    bridges: tuple[CorrectionBridge, ...]

    @property
    def output(self) -> torch.Tensor:
        return self.consumed_post_acts[-1]


class InstrumentedMLP:
    """Detached float snapshot with reusable per-layer geometry.

    Supports chains of Linear/ReLU, optionally ending with a linear output
    or ReLU. Rejects other modules instead of silently changing semantics.
    Float32 and float64 are supported; no activation or bias quantization.
    """

    def __init__(self, model: nn.Module, *, quantizer: WeightQuantizer | None = None,
                 pinv_rtol: float | None = None):
        sequence = model if isinstance(model, nn.Sequential) else getattr(model, "layers", None)
        if not isinstance(sequence, nn.Sequential):
            raise ValueError("model must be nn.Sequential or expose Sequential layers")
        if type(sequence) is not nn.Sequential:
            raise ValueError("custom Sequential forward implementations are not supported")
        if any(hasattr(module, "qconfig") or hasattr(module, "weight_fake_quant")
               for module in sequence.modules()):
            raise ValueError("use an unprepared float network, not a QAT/PTQ model")
        if any(module._forward_hooks or module._forward_pre_hooks for module in sequence.modules()):
            raise ValueError("forward hooks may change semantics; use a plain float snapshot")
        # Only these two structures have a known forward contract here.
        if not isinstance(model, nn.Sequential):
            from aleph.models import MLP
            if type(model) is not MLP or not isinstance(model.quant, torch.ao.quantization.QuantStub) or not isinstance(model.dequant, torch.ao.quantization.DeQuantStub):
                raise ValueError("only an unconverted aleph.models.MLP wrapper is supported")
            if hasattr(model, "qconfig"):
                raise ValueError("use an unprepared float MLP, not a QAT model")
        self.quantizer = quantizer or WeightQuantizer()
        self.names, self.weights, self.biases, self.activations = [], [], [], []
        modules = list(sequence.named_children())
        cursor = 0
        while cursor < len(modules):
            name, layer = modules[cursor]
            if type(layer) is not nn.Linear:
                raise ValueError("expected a Linear followed by ReLU or the end of the chain")
            self.names.append(name)
            self.weights.append(layer.weight.detach().T.contiguous().clone())
            self.biases.append(None if layer.bias is None else layer.bias.detach().clone())
            cursor += 1
            relu = cursor < len(modules) and type(modules[cursor][1]) is nn.ReLU
            self.activations.append("relu" if relu else "identity")
            cursor += int(relu)
            if cursor < len(modules) and not relu:
                raise ValueError("hidden Linear layers must be followed by ReLU")
        if not self.weights:
            raise ValueError("model must contain at least one Linear")
        for previous, current in zip(self.weights, self.weights[1:]):
            if previous.shape[1] != current.shape[0]:
                raise ValueError("adjacent layer widths do not match")
            if previous.dtype != current.dtype or previous.device != current.device:
                raise ValueError("all layers must share dtype and device")
        self.quantized_weights = [self.quantizer(W) for W in self.weights]
        self.decompositions = [MapDecomposition.from_weights(W, q.quantized, rtol=pinv_rtol)
                               for W, q in zip(self.weights, self.quantized_weights)]
        self.float_decompositions = [MapDecomposition.from_weights(W, W, rtol=pinv_rtol)
                                    for W in self.weights]

    def _activate(self, value: torch.Tensor, index: int) -> torch.Tensor:
        return torch.relu(value) if self.activations[index] == "relu" else value

    @torch.no_grad()
    def run(self, x: torch.Tensor, *, quantized_layers: Iterable[int] | None = None,
            bridges: Iterable[CorrectionBridge] = ()) -> ActivationTrace:
        count = len(self.weights)
        selected = set(range(count)) if quantized_layers is None else set(quantized_layers)
        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count for i in selected):
            raise ValueError("quantized layer indices must be integers within the network")
        bridges = tuple(bridges)
        for bridge in bridges:
            if bridge.target >= count or bridge.source not in selected:
                raise ValueError("bridge target must exist and source must be quantized")
        if len({b.target for b in bridges}) != len(bridges):
            raise ValueError("only one exact bridge per target is supported")
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.weights[0].shape[0]:
            raise ValueError("x must be a nonempty [samples, input] matrix")
        if x.dtype != self.weights[0].dtype or x.device != self.weights[0].device or not torch.isfinite(x).all():
            raise ValueError("x must be finite and share model dtype and device")
        inputs, pre, post = [], [], []
        a = x.detach().clone()
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            inputs.append(a)
            z = a @ W + (0 if b is None else b)
            a = self._activate(z, i)
            pre.append(z)
            post.append(a)
        fp = ForwardTrace(pre, post)
        oracle = ResidualOracle(fp)
        targets = {bridge.target for bridge in bridges}
        raw_pre, raw_post, consumed, layers = [], [], [], []
        a = x.detach().clone()
        for i in range(count):
            decomp = self.decompositions[i] if i in selected else self.float_decompositions[i]
            ledger = ErrorLedger.compute(a, decomp, bias=self.biases[i],
                                         activation=lambda z, index=i: self._activate(z, index))
            h = ledger.quant_post
            residual = oracle.residual(h, i)
            corrected = oracle.correct(h, i) if i in targets else h
            propagated = ledger.fp_post - fp.post_acts[i]
            total = h - fp.post_acts[i]
            tolerance = 1e-5 if x.dtype == torch.float32 else 1e-10
            torch.testing.assert_close(ledger.total + propagated, total, atol=tolerance, rtol=tolerance)
            mask = ((ledger.quant_pre > 0) != (fp.pre_acts[i] > 0)) if self.activations[i] == "relu" else None
            layers.append(LayerTrace(i, self.names[i], self.activations[i], i in selected,
                                     self.quantized_weights[i] if i in selected else None,
                                     a, inputs[i], fp.pre_acts[i], fp.post_acts[i],
                                     ledger.quant_pre, h, corrected, residual, i in targets,
                                     ledger, propagated, total, mask))
            raw_pre.append(ledger.quant_pre)
            raw_post.append(h)
            consumed.append(corrected)
            a = corrected
        return ActivationTrace(fp, ForwardTrace(raw_pre, raw_post), consumed, layers, bridges)
