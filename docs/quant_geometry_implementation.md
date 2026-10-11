# Quantization geometry: implementation and reproduction

The [research note](quant_geometry_foundations.md) defines the objects and their geometric meaning. This document describes the current package, its conventions, and how to reproduce the artifacts. Verification evidence is in the [verification report](quant_geometry_verification.md); work status is in the [progress tracker](quant_geometry_progress.md).

## Quantizer and decomposition

`aleph/qgeom/weight_quantizer.py` adds a deterministic, per-tensor symmetric weight quantizer. It returns detached original and dequantized weights plus bit width, scale, bounds and rounding metadata. For `bits=b`, the integer bounds are `[-(2**(b-1)-1), +(2**(b-1)-1)]`; one signed code is unused to keep the grid symmetric. Scale is `absmax/qmax`; a zero matrix uses scale 1. Rounding is nearest with ties to even. The runner leaves biases and activations floating point. This is inspectable fake quantization, not packed integer execution.

`MapDecomposition.from_weights(W, Wq, rtol=...)` uses row-vector weights `[input, output]`. It returns R/U, reconstruction residual, numerical rank, retained-spectrum conditioning, and singular spectra. The default pseudoinverse cutoff is `max(shape)*dtype_epsilon` times the largest singular value. `U_rank` uses an absolute threshold scaled to the original maps. Full-map `Wq.T @ U` need not vanish when nonzero singular values are truncated; see the research note's numerical-policy qualification.

## Ledger interface

`ErrorLedger.compute(x, decomposition, bias=..., activation=...)` returns `uncomp`, `nonlinear`, `range`, and Q-minus-FP `total`, plus the states `quant_pre`, `fp_pre`, `quant_post`, `fp_post`, `recoverable_pre`, `recoverable_post`, and `transported_post`. It checks affine reconstruction and component telescoping coordinate by coordinate. Default absolute and relative tolerances are `1e-5` for float32 and `1e-10` for float64; both can be overridden. Ill-conditioned cases may need float64 or an explicitly chosen rank policy.

`ledger.gram()` returns the mean inner-product matrix of the three components, ordered uncompensated/nonlinear/range. It is an uncentered second moment, not covariance or an allocation into additive percentages.

## Shared MLP traces and residual oracle

`aleph/qgeom/instrumented_mlp.py` extracts the reusable forward/oracle responsibilities previously embedded in `notebooks/canonical_error_correction.py`. It reuses `ForwardTrace` from `aleph/qgeom/canonical.py`; it does not replace the existing canonical-coordinate analysis. The new runner adapts the old oracle concept to explicit post-activation FP-minus-Q residuals, preserving the raw damaged states.

`InstrumentedMLP` snapshots an ordinary float `nn.Sequential` chain of Linear/ReLU modules or an unprepared `aleph.models.MLP`. Hidden linear layers require ReLU; the final layer may be linear or ReLU. Other modules, prepared QAT models and unsupported wrappers are rejected. PyTorch `[output,input]` weights are transposed at the adapter boundary. The snapshot is detached, independent of later edits to the model, and supports float32/float64 on the model's device.

Each `LayerTrace` contains:

- Layer index/name, activation, quantization metadata and selected status.
- Actual run input and reference FP input.
- Reference FP and raw Q pre/post states; consumed post state after any correction.
- Exact oracle residual, correction-applied flag and ReLU disagreement mask against reference FP.
- Same-input creation ledger, propagated vector and raw total vector.

For a run input `a` that may already differ from the FP input, the runner defines local creation as `phi(aWq+b)-phi(aW+b)` and propagated error as `phi(aW+b)-h_fp`. These vectors telescope to raw total error. Norms do not necessarily add. This is a local-versus-incoming accounting identity; it does not separately propagate the three mechanisms through nonlinear downstream layers.

`ActivationTrace.fp` is the reference `ForwardTrace`; `.raw_quantized` contains pre/post states before each intervention; `.consumed_post_acts` contains states passed to the next layer. A raw trace in an oracle run is conditioned on earlier corrected inputs, so it must not be confused with ordinary quantized inference. `.output` is the final consumed state.

`ResidualOracle.residual(hq, layer)` returns `h_fp-hq`, the **opposite sign** to the ledger. A `CorrectionBridge(source=i,target=i)` restores the output of linear layer `i` before layer `i+1` consumes it. A later target restores at that later layer. Indices are zero-based **linear-layer indices**, not Sequential module positions. An exact bridge restores all accumulated discrepancy at its target; it is a source-specific intervention only when the named source is the sole perturbed layer.

The [verification report](quant_geometry_verification.md#diagnostic-figures-and-numerical-evidence) records the runner and oracle diagnostic figures and measured restoration results.

### API example

```python
import torch
from torch import nn
from aleph.qgeom import (
    WeightQuantizer, MapDecomposition, ErrorLedger,
    InstrumentedMLP, CorrectionBridge,
)

# Geometry modules use row-vector convention.
W = torch.tensor([[1., -.35], [.28, .22]], dtype=torch.float64)
x = torch.tensor([[1., 2.]], dtype=W.dtype)
quantizer = WeightQuantizer(bits=2)
q = quantizer(W)
decomp = MapDecomposition.from_weights(W, q.quantized)
ledger = ErrorLedger.compute(x, decomp)
assert torch.allclose(ledger.uncomp + ledger.nonlinear + ledger.range, ledger.total)

# Runner adapts nn.Linear weights and keeps all biases/activations float.
model = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 2))
runner = InstrumentedMLP(model, quantizer=quantizer)
ordinary = runner.run(x.float(), quantized_layers=[0])
corrected = runner.run(x.float(), quantized_layers=[0],
                       bridges=[CorrectionBridge(source=0, target=0)])
```

## Reproduce the artifacts

Run from the repository root:

```sh
MPLCONFIGDIR=/tmp/qgeom-mpl XDG_CACHE_HOME=/tmp/qgeom-cache .venv/bin/python -m experiments.quant_geometry_foundations
MPLCONFIGDIR=/tmp/qgeom-mpl XDG_CACHE_HOME=/tmp/qgeom-cache .venv/bin/python -m experiments.quant_geometry_visual_story
MPLCONFIGDIR=/tmp/qgeom-mpl XDG_CACHE_HOME=/tmp/qgeom-cache .venv/bin/python -m experiments.quant_geometry_ledger_figures
```

The first command produces eight diagnostic figures and the run manifest. The second produces three geometric storyboards and exact endpoint data. The third draws the exact state comparisons that define each ledger component and a signed cancellation path. Outputs are under `docs/figures/quant_geometry/`.

To populate the animation template, add `--fragment /absolute/path/story.html` to the second command. Its source is `animations/quant_geometry_story.html`, and its geometry is supplied by `aleph/qgeom/geometry_story.py`. Playback interpolates corresponding points between exact endpoint states; intermediate frames are explanatory motion, not additional bit widths or inference states. The payload rounds endpoint coordinates to six decimals for display. The static figures use the float64 tensors directly.
