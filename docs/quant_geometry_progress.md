# Quantization geometry progress

Plan: [quantization_geometry_plan.md](quantization_geometry_plan.md)

Started: 2026-10-11. Branch: `codex/quant-geometry-framework`.
Baseline: `0be49df7` (`Snapshot research state before quantization return`).

## Tracking rules

- Status: TODO, IN PROGRESS, PROTOTYPE, DONE, or BLOCKED.
- PROTOTYPE means related code exists but does not meet the plan's acceptance criteria.
- Mark DONE only with reusable implementation, relevant passing tests, and required experiment artifacts. Record paths and verification commands below.
- For each experiment record revision, seed, dataset/split, model dimensions, quantization scheme/scales/bits, selected quantized layers, correction sites, tolerances, and output paths.
- Update this file with each meaningful implementation change. Keep research findings separate from implementation completion; append decisions and verification to the dated log.

## Repository audit (initial baseline, before implementation)

| Capability | Existing evidence | Assessment / remaining work |
|---|---|---|
| Weight-only quantization | `aleph/qgeom/core.py::quantize`; `notebooks/canonical_error_correction.py::extract_weights` | Nearest-grid rounding exists in NumPy and Torch. No bounded bit-width weight-only API with original/quantized weights and metadata. Notebook grid rounding does not itself enforce a bit budget. |
| General quantization | `aleph/quantization.py` | Fake quantization, rounding modes, 2/4/8-bit ranges and QAT exist. Reuse arithmetic selectively; QAT and activation calibration are outside this phase. |
| Map decomposition | `aleph/qgeom/canonical.py::CanonicalSpaceTracker` | Cumulative-map pseudoinverse exists, but is not the plan's per-layer `W = Wq R + U`. Implement the latter independently. |
| Error attribution | `aleph/qgeom/canonical.py::error_attribution` | Reusable per-sample local/propagated preactivation errors exist. Three post-activation mechanism vectors and numerical telescoping checks are missing. |
| Alternate geometric split | `notebooks/canonical_learned_correction.py::geometric_error_decomposition` | ReLU agreement/disagreement split exists; it is not the recoverable/nonlinear/range ledger. |
| Traces / ReLU states | `aleph/qgeom/canonical.py::ForwardTrace`, `ReLUDisagreementTracker` | Reusable pre/post tensor containers and mask diagnostics exist. Forward runner is notebook-local; corrected/raw states and layer metadata need adding. |
| ReLU MLP / datasets | `aleph/models.py::MLP`; `aleph/datasets.py`; canonical notebooks | Reuse float model and synthetic data. Add a runner for independently selected weight-quantized layers, retaining float biases and activations. |
| Exact oracle / selected sites | `notebooks/canonical_error_correction.py::PerfectCorrection` | Weight-only oracle prototype corrects preactivation and supports selected sites. Extract/adapt to explicit post-activation residual injection; retain raw states before correction. |
| Existing model correction | `aleph/models.py::MLPWithCorrection` | Oracle and learned correction methods exist, but these inference methods quantize activations. They do not satisfy the new weight-only protocol. |
| Low-rank / learned correction | `notebooks/canonical_learned_correction.py`; `notebooks/correction_distillation.py` | SVD oracle projections and learned correction prototypes exist. A projection using FP residuals is not a fitted correction using only quantized representations. Need shared families and held-out evaluation. |
| Reports / metrics / policies | `aleph/qgeom/core.py`, `metrics.py`, `policy.py` | Reusable schemas, spectra/rank-related metrics, ReLU diagnostics and correction-site scoring exist. Correctability is a heuristic; policy simulation uses supplied evaluators/gains, not a built-in causal intervention runner. |
| Visualization | `aleph/qgeom/drawing_2d.py`, `drawing_3d.py`; canonical notebooks | Reuse plotting support and examples. New ledger and recoverability plots still needed. |
| Source propagation / interactions | No dedicated implementation found in inspected `aleph/qgeom` and canonical notebooks | Need isolated-source counterfactuals, ordered telescoping, and Gram accounting. Canonical projection through products of weights is not an exact nonlinear propagation model. |

## Milestones and work queue

| ID | Status | Deliverable / completion gate |
|---|---|---|
| A0 | DONE | Read plan, audit relevant reusable code and prototypes, create branch and this tracker. |
| M1 | DONE | `weight_quantizer.py`, `map_decomposition.py`, `ledger.py`; exactness/orthogonality tests, bounded bit grid, bias-aware per-sample vectors; Figures 1–5 in the foundations report. |
| M2 | DONE | `instrumented_mlp.py` reuses `ForwardTrace`, selects arbitrary weight-quantized layers, records FP/raw pre/post and corrected consumed post states plus metadata; local/propagated identity tested; Figure 6. |
| M3 | DONE | Shared `ResidualOracle` and `CorrectionBridge`; immediate intervention tested independently at every layer of a variable-width MLP; raw damage preserved and FP downstream trajectory restored; Figure 6. |
| M4 | DONE | All-layer immediate mode tested: inputs and consumed outputs match FP at every layer while local damage remains visible; comparison with ordinary Q shown in Figure 7. This is apparatus validation on an untrained MLP, not a trained-model research result. |
| M5 | PROTOTYPE | Existing selected-site oracle and SVD experiments are ingredients only. Isolate one source layer, delay injection, fit linear/affine corrections on calibration data, evaluate held-out residuals/ranks/amplification versus depth. |
| L1 | TODO | Ordered component counterfactual propagation with explicit ordering and exact downstream telescoping. |
| L2 | PROTOTYPE | Local `ErrorLedger.gram()` and Figure 5 now expose interaction; downstream recoverability curves and tolerance-based horizon still TODO. |

The first three requested next steps are complete: reusable quantizer/decomposition/ledger, exact tests and single-layer figures, and shared MLP tracing/oracle extraction. Core code stays under the installed `aleph/qgeom` package.

Next implementation slice:

1. Run a trained 2D classification MLP with ordinary inference and immediate oracle; document per-layer component distributions and interactions alongside task accuracy.
2. Implement M5 constrained linear/affine families fitted on calibration data and evaluated on held-out data; plot delayed recoverability and correction complexity.
3. Add ordered component propagation before interpreting mechanism-specific downstream effects. Proceed to MNIST or transformers only after these protocols are validated.

## Mathematical / protocol decisions

- Implemented row-vector weights `[in, out]` internally; the MLP adapter transposes `nn.Linear.weight` at the boundary.
- Resolved float bias with affine transport `A(v)=vR+b-bR`. Use uncompensated `h-A(h)`, nonlinear `A(h)-phi(A(zq))`, and range `phi(A(zq))-y`. This preserves exactness and zero nonlinear error for the identity activation. The zero-bias case reduces to the written plan.
- Ledger error is Q minus FP; oracle residual is FP minus Q. Name these signs explicitly.
- Full row rank of a widening `Wq` gives `U=0` in exact arithmetic if no nonzero singular values are discarded. Under truncation, `U` is a correction-policy remainder and need not be orthogonal to the full `Wq` or prove lost information. See the mathematical audit and Figure 8 in the foundations report.
- A same-input ledger measures local creation. Comparing layers fed different upstream states also includes propagation and needs separate accounting.
- Components are not generally orthogonal. Their norm/energy ratios are not additive percentages of total error; report cross terms or use clearly defined allocations.
- Attribution also depends on the declared extension of `R`: a change `R -> R+N` with `Wq N=0` can alter post-ReLU components while preserving the original linear map and total error. The implementation fixes the identity-centered convention.
- An exact additive oracle can restore FP states after ReLU. The nonlinear ledger term measures failure of the particular linear correction to commute with ReLU, not proof that every correction family fails.
- Recoverability horizons require a named correction family, held-out evaluation and tolerance. Error may become easier to recover again at later bottlenecks; report full curves alongside any first-threshold-crossing statistic.

## Verification and dated log

### 2026-10-11 — audit and tracking setup

- Created branch from `codex/quantization-revival` at `0be49df7`; preserved existing untracked plan `docs/quantization_geometry_plan.md`.
- Inspected relevant package modules, tests and canonical/correction notebook sources. Existing notebook result claims were not revalidated by retraining.
- Baseline command: `.venv/bin/python -m pytest tests/test_qgeom_metrics.py tests/test_qgeom_policy.py -q`: **8 passed** in 2.22 seconds. These cover existing metrics and policy selection, not the new plan's identities or oracle.
- No implementation milestones certified complete by this audit. Runtime coverage of map decomposition, the new ledger and the weight-only oracle still needs to be established.

### 2026-10-11 — foundations, figures and shared runner

- Added bounded weight-only quantization, map decomposition, affine bias-aware exact ledger, and a local Gram interaction summary under `aleph/qgeom`.
- Adapted notebook-local forward/oracle ideas into a detached MLP snapshot and post-activation residual bridge API, reusing the existing `ForwardTrace`. Original notebook implementations are preserved.
- Added 36 new acceptance tests; targeted suite including existing geometry and model tests: **65 passed**. Command: `.venv/bin/python -m pytest tests/test_qgeom_exact_ledger.py tests/test_qgeom_instrumented_mlp.py tests/test_qgeom_metrics.py tests/test_qgeom_policy.py tests/test_models.py -q`.
- Reproduction command: `.venv/bin/python -m experiments.quant_geometry_foundations`. Generated seven PNG figures and `docs/figures/quant_geometry/run_manifest.json`, including exact weights/biases and measured results.
- Float64 ledger maximum telescoping residual: `5.55e-17`. Float32 all-immediate example output discrepancy: `0.0`; acceptance tests use numerical tolerances.
- Explanations, equations, example vectors, source provenance, API usage and a contribution-to-figure/test mapping are in [quant_geometry_foundations.md](quant_geometry_foundations.md). Every named mechanism has its own figure (Figures 2–4); decomposition, accounting and runner/oracle steps also have figures.
- Visually inspected all seven generated figures. The saved manifest records the pre-commit base revision and dirty state; implementation and artifacts are preserved together in the branch's foundations checkpoint.
- M1–M4 acceptance gates pass. M4 validation follows naturally from the extracted runner; no constrained recoverability or trained-model conclusions are claimed.

### 2026-10-11 — documentation mathematics audit

- Verified reconstruction, affine-bias handling, telescoping, Gram-energy identity, the `(1,2)` example and local/propagated accounting. Added explicit derivations and dimensions to the report.
- Corrected interpretation of tolerance-truncated pseudoinverses: orthogonality is only to the retained span; a discarded nonzero direction can remain recoverable by a high-gain decoder. Added Figure 8 and its separate prescribed-matrix manifest fixture.
- Clarified that individual components are counterfactual differences, not each Q-minus-FP, and depend on the chosen extension of `R`; only the total is fixed by FP/Q outputs. Added a zero-quantization-error counterexample with nonzero cancelling components under an alternative correction.
- Corrected Figure 3's explanation to cover both signs of the first input coordinate, including its selected negative-input example; documented the exact fixture formula `e_NL=(0,-0.35*abs(s))`.
- Added five independent mathematical regression tests. Targeted suite: **70 passed** using the same command recorded above. Regenerated eight figures, checked the new audit figure visually, and preserved the original numerical results.
