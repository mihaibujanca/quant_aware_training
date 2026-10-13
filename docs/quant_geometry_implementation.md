# Quantization geometry: implementation and reproduction

The [research note](quant_geometry_foundations.md) defines the objects and their geometric meaning. This document describes the current package, its conventions, and how to reproduce the artifacts. Verification evidence is in the [verification report](quant_geometry_verification.md); work status is in the [progress tracker](quant_geometry_progress.md).

## Quantizer and decomposition

`aleph/qgeom/weight_quantizer.py` adds a deterministic, per-tensor symmetric weight quantizer. It returns detached original and dequantized weights plus bit width, scale, bounds and rounding metadata. For `bits=b`, the integer bounds are `[-(2**(b-1)-1), +(2**(b-1)-1)]`; one signed code is unused to keep the grid symmetric. Scale is `absmax/qmax`; a zero matrix uses scale 1. Rounding is nearest with ties to even. The runner leaves biases and activations floating point. This is inspectable fake quantization, not packed integer execution.

`MapDecomposition.from_weights(W, Wq, rtol=...)` uses row-vector weights `[input, output]`. It returns R/U, reconstruction residual, numerical rank, retained-spectrum conditioning, and singular spectra. The default pseudoinverse cutoff is `max(shape)*dtype_epsilon` times the largest singular value. `U_rank` uses an absolute threshold scaled to the original maps. Full-map `Wq.T @ U` need not vanish when nonzero singular values are truncated; see the research note's numerical-policy qualification.

R and U are computed and retained in W's floating-point dtype (float32 or float64). R is not quantized. Wq also uses that dtype to store dequantized low-bit grid values; its floating storage does not give its coefficients the unconstrained precision of R. Quantizing R would define a different correction family and could leave residual error even for square invertible Wq.

## Ledger interface

`ErrorLedger.compute(x, decomposition, bias=..., activation=...)` returns `uncomp`, `nonlinear`, `range`, and Q-minus-FP `total`, plus the states `quant_pre`, `fp_pre`, `quant_post`, `fp_post`, `recoverable_pre`, `recoverable_post`, and `transported_post`. It checks affine reconstruction and component telescoping coordinate by coordinate. Default absolute and relative tolerances are `1e-5` for float32 and `1e-10` for float64; both can be overridden. Ill-conditioned cases may need float64 or an explicitly chosen rank policy.

The implementation stores Q-minus-FP vectors. The research note's [forward repair account](quant_geometry_foundations.md#b3-float-bias-and-the-implementation-sign-convention) uses their negatives: `uncomp = -c_R`, `nonlinear = -c_NL`, `range = -c_U`, and `total = -c_total`. The available map correction is c_R; c_NL and c_U explain its remaining balance. Exported keys and historical filenames are preserved. The Gram matrix is unchanged when all vector signs are reversed.

`ledger.gram()` returns the mean inner-product matrix of the three components, ordered uncompensated/nonlinear/residual deformation. It is an uncentered second moment, not covariance or an allocation into additive percentages.

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
MPLCONFIGDIR=/tmp/qgeom-mpl XDG_CACHE_HOME=/tmp/qgeom-cache .venv/bin/python -m experiments.quant_geometry_scalar_measurement
MPLCONFIGDIR=/tmp/qgeom-mpl XDG_CACHE_HOME=/tmp/qgeom-cache .venv/bin/python -m experiments.quant_geometry_rectangular_measurement
MPLCONFIGDIR=/tmp/qgeom-mpl XDG_CACHE_HOME=/tmp/qgeom-cache .venv/bin/python -m experiments.quant_geometry_visual_story
MPLCONFIGDIR=/tmp/qgeom-mpl XDG_CACHE_HOME=/tmp/qgeom-cache .venv/bin/python -m experiments.quant_geometry_ledger_figures
MPLCONFIGDIR=/tmp/qgeom-mpl XDG_CACHE_HOME=/tmp/qgeom-cache .venv/bin/python -m experiments.quant_geometry_reconstruction_comparison
```

The foundations command produces eight diagnostic figures and the run manifest. The scalar-measurement command produces the weight-direction/fiber figure and `scalar_measurement_fixture.json`, with the exact weights, repair gain, residual and A/B responses. The rectangular-measurement command produces the 3→2 measured-plane/fiber figure and `rectangular_measurement_fixture.json`, recording both ranks, the nonzero Q singular values, and exact A/B states. The visual-story command produces three geometric storyboards and exact endpoint data. The ledger-figure command draws the forward Q→FP repair account (Figures 4–7), the incoming/local/oracle balance (Figure 8), and `repair_ledger_figure_fixture.json` with exact endpoints and signed contributions. Historical `ledger_geometry_uncomp` / `ledger_geometry_nonlinear` / `ledger_geometry_range` filenames now contain the forward c_R / c_NL / c_U explanation. The reconstruction-comparison command produces the two-row/four-column `geometry_repair_vs_reconstruction.png` and `repair_vs_reconstruction_fixture.json`. It records all three repairs, the uncorrected baseline, all three objective values, and the input geometry. Outputs are under `docs/figures/quant_geometry/`.

The comparison uses the full-rank 3→2 fixture from `build_rectangular_measurement_story`. The 2D grid/circle parameters specify a tilted 3D input sheet; offsets of −0.15, 0, and +0.15 along the Q blind direction give 2,577 calibration samples with rank-3 input data. All samples have equal weight, including duplicate intersections; named A/B probes are excluded on all three sheets. The pictures show the central sheet; reported losses use the whole cloud. Map, preactivation data, and post-ReLU data fits all use the same full-precision 2×2 correction family.

The script checks each fit's normal equation, each optimizer's strict preference under its own objective, the analytic preactivation fit and its agreement with FP on the displayed central sheet, and the second-moment weighting identity. A separate isotropic preactivation fit recovers the map repair. This is a deterministic explanatory fixture, not a held-out correction experiment or a new runner API.

Research Figures 1, 2, 4, and 5 share W, Wq, R, the grid/circle and A/B inputs. The `deformation` and `relu` story keys select different stages of this one example; the rank-collapse story is a separate fixture. The network/oracle figure has its own propagation fixture.

To populate the animation template, add `--fragment /absolute/path/story.html` to the visual-story command. Its source is `animations/quant_geometry_story.html`, and its geometry is supplied by `aleph/qgeom/geometry_story.py`. Playback interpolates corresponding points between exact endpoint states; intermediate frames are explanatory motion, not additional bit widths or inference states. The payload rounds endpoint coordinates to six decimals for display. The static figures use the float64 tensors directly.
