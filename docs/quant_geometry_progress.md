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
- Write Markdown math with `$...$` inline and `$$...$$` display blocks for VS Code preview compatibility. Check new formulas against KaTeX rather than assuming interface-specific rendering.
- Teaching fixtures must visibly move the input under quantization; do not use identity Wq as a geometric explanation. Show the expected output alongside the failed route.
- Each tracked repair/balance contribution needs a formal definition and a picture of its exact endpoints. Research arrows point forward Q→FP. Start with the observed correction owed, apply the available repair, and explain its outstanding balance; do not introduce three peer error sources or lead with state indices.
- Explanatory figures must show the geometric cause with identifiable samples or shapes across transformations. Norm curves, heatmaps and matrix plots support the reference; they do not by themselves satisfy the intuition requirement.
- Keep the main research narrative on the general map/repair/activation/ledger argument. Put specialized measurement examples and technical machinery after that argument; a clarification should not automatically become another interlude in Section 2.
- Introduce each operation geometrically and define its notation before using it in an objective, figure, or ledger. In particular, explain and define ReLU before post-ReLU reconstruction; a symbol declaration in the opening does not supply the conceptual introduction.
- Distinguish calibration-conditioned reconstruction from the geometric map repair used for attribution. Record which objective is being optimized and the data weighting; neither objective substitutes for the other.
- Compare map, preactivation data, and post-activation data objectives explicitly. Show the input weighting and the geometric outcomes; hold the dataset and correction family fixed when isolating activation's effect. An exactly repairable fixture can hide the map-versus-preactivation-data distinction.
- Keep the square-invertible exact-repair guarantee and the prevalence of rectangular ANN layer maps explicit beside the main repair objective. State that R uses W's full floating-point precision and is not quantized; do not confuse Wq's dequantized storage dtype with its coefficient grid.
- When introducing a new ledger term, identify the assumption being relaxed, the states the existing account compares, and the extra displacement needed to reach the actual FP target. A residual definition alone does not explain its role.

## Repository audit (initial baseline, before implementation)

| Capability | Existing evidence | Assessment / remaining work |
|---|---|---|
| Weight-only quantization | `aleph/qgeom/core.py::quantize`; `notebooks/canonical_error_correction.py::extract_weights` | Nearest-grid rounding exists in NumPy and Torch. No bounded bit-width weight-only API with original/quantized weights and metadata. Notebook grid rounding does not itself enforce a bit budget. |
| General quantization | `aleph/quantization.py` | Fake quantization, rounding modes, 2/4/8-bit ranges and QAT exist. Reuse arithmetic selectively; QAT and activation calibration are outside this phase. |
| Map decomposition | `aleph/qgeom/canonical.py::CanonicalSpaceTracker` | Cumulative-map pseudoinverse exists, but is not the plan's per-layer `W = Wq R + U`. Implement the latter independently. |
| Error attribution | `aleph/qgeom/canonical.py::error_attribution` | Reusable per-sample local/propagated preactivation errors exist. Three post-activation mechanism vectors and numerical telescoping checks are missing. |
| Alternate geometric split | `notebooks/canonical_learned_correction.py::geometric_error_decomposition` | ReLU agreement/disagreement split exists; it is not the uncompensated/nonlinear/residual-deformation ledger. |
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
| M1 | DONE | `weight_quantizer.py`, `map_decomposition.py`, `ledger.py`; exactness/orthogonality tests, bounded bit grid, bias-aware per-sample vectors; diagnostic Figures 1–5 in the verification report. |
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
- Resolved float bias with affine transport `A(v)=vR+b-bR`. The forward account uses c_R=A(h)-h, c_NL=phi(A(zq))-A(h), and c_U=y-phi(A(zq)). Their negatives are the stored implementation fields. This preserves exactness and zero activation-order balance for identity activation.
- The research ledger reconciles FP minus Q, the correction still owed. The implementation retains Q-minus-FP fields; their negative mapping is in Technical Details. The oracle adds the forward total correction.
- Full row rank of a widening `Wq` gives `U=0` in exact arithmetic if no nonzero singular values are discarded. Under truncation, `U` is a correction-policy remainder and need not be orthogonal to the full `Wq` or prove lost information. See the mathematical audit and Figure 8 in the verification report.
- A same-input ledger measures local creation. Comparing layers fed different upstream states also includes propagation and needs separate accounting.
- Components are not generally orthogonal. Their norm/energy ratios are not additive percentages of total error; report cross terms or use clearly defined allocations.
- Attribution also depends on the declared extension of `R`: a change `R -> R+N` with `Wq N=0` can alter post-ReLU components while preserving the original linear map and total error. The implementation fixes the identity-centered convention.
- An exact additive oracle can restore FP states after ReLU. The activation-order balance compares the chosen map repair on opposite sides of ReLU; a nonzero value does not prove that every other correction family fails.
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

### 2026-10-11 — replace formula-first exposition with geometric motion

- Reviewed the Manim source/narrative and a rendered frame from `LayerProgressionScene`. Adopted its continuous shape/point correspondence, rather than presenting scalar diagnostics as the explanation.
- Added [Watch the shape, then name the error](quant_geometry_visual_guide.md) as the workstream's entry point. Relabeled the foundations report as a technical reference and linked the guide from the root README and animation documentation.
- Added three geometric stories under `aleph/qgeom/geometry_story.py`: invertible deformation with U=0, information erased by ReLU despite an invertible quantized map, and a true rank-1 collapse that R alone cannot widen. Every story uses the existing quantizer/decomposition/ledger, zero bias, a grid/circle and named probes A/B.
- Added three reproducible storyboards, exact endpoint data, and `animations/quant_geometry_story.html` for animated point-preserving transitions. Kept coordinate scales fixed throughout each story. Distinguished exact inference endpoints from visual interpolation frames.
- Added four tests for the claims in the pictures, including A/B collisions versus distinct FP targets and restoration by input-dependent xU. Targeted suite including existing ledger/runner/metrics/policy/model tests: **74 passed**.
- Reproduction: `MPLCONFIGDIR=/tmp/qgeom-mpl .venv/bin/python -m experiments.quant_geometry_visual_story`. Add `--fragment /absolute/path/story.html` to populate the moving version.
- Browser review verified step controls, full sequence playback, zero JavaScript errors, ReLU collision label, and a 360px layout without horizontal overflow. Visually inspected all three storyboards and desktop/mobile animation screenshots.
- This corrects the explanatory layer; no quantization or oracle algorithm was changed, and no new trained-model result is claimed.

### 2026-10-11 — unite geometry and rigor; separate document roles

- Rewrote [the foundations research note](quant_geometry_foundations.md) around the affine map split, four precisely defined counterfactual states, and the exact local ledger. Paired each formal component with an actual state-pair figure and signed A/B arrows. Retained bias preservation, numerical truncation, extension dependence, telescope proof and Gram interaction identity.
- Added `experiments/quant_geometry_ledger_figures.py` and four reproducible figures: three exact component comparisons and the independently specified `(1,2)` cancellation path. Pre-activation storyboards explain causes; new figures show the activated quantities actually measured.
- Moved API details and reproduction into [the implementation README](quant_geometry_implementation.md). Moved numerical evidence, diagnostic figures, contribution mapping, and the mathematical audit into [the verification report](quant_geometry_verification.md). The old visual guide is now navigation to the unified research note.
- Reproduction: `MPLCONFIGDIR=/tmp/qgeom-mpl XDG_CACHE_HOME=/tmp/qgeom-cache .venv/bin/python -m experiments.quant_geometry_ledger_figures`. Visually inspected all four new figures and checked caption coordinates against the existing ledger states.
- No core algorithm changed. The targeted acceptance suite remains **74 passed**. Research scope and the next implementation slice remain unchanged.

### 2026-10-11 — Figure 1: make quantization and correction distinguishable

- Replaced the identity-quantized deformation fixture with W=[[1,0.65],[-0.35,0.8]], quantized at 2 bits to Wq=[[1,1],[0,1]]. Both FP and Q visibly deform the input; their shapes differ. R=[[1.35,-0.15],[-0.35,0.8]] recovers the FP map with U=0.
- Removed the Figure 1 reference overlays and ambiguous probe arrows. Numbered panels, named their actual operations, and explicitly identified the matching right panels as the same endpoint reached by different routes. Replaced “weights snapped” with “apply rounded weights” and documented coefficient rounding.
- Regenerated endpoint data and the moving fragment. Regenerated the dependent Figure 3 uncompensated comparison and updated its exact A/B coordinates. Other figure examples are unchanged.
- Strengthened the existing deformation test with independently specified quantized weights, R, and probe coordinates, plus checks that the Q shape differs from both input and FP. Targeted acceptance suite: **74 passed**. Visually inspected revised Figures 1 and 3; `git diff --check` passes.

### 2026-10-11 — VS Code math rendering compatibility

- Replaced LaTeX-style backslash math delimiters in the research note with dollar delimiters supported by VS Code's built-in Markdown preview. No equations changed.
- Checked all 17 display equations and 55 inline expressions using the KaTeX renderer bundled with the installed VS Code, with parse errors and strict warnings treated as failures. All render without errors; delimiters are balanced.

### 2026-10-11 — remove identity from the ReLU explanation

- Replaced the remaining identity-quantized teaching example with W=[[1,-0.4],[0.65,0.85]], Wq=[[1,0],[1,1]], R=[[1,-0.4],[-0.35,1.25]], U=0. The quantized shear moves both probes; correcting before clipping produces distinct nonzero FP outputs (0,0.235) and (0,0.07), whereas clipping first merges them at zero.
- Regenerated the ReLU storyboard, exact nonlinear state-pair figure, and animation endpoint data; documented matrices and both nonzero signed vectors. Separated the two overlapping nonlinear arrows visually while retaining their exact endpoints.
- Strengthened independently specified coordinate checks and added a check that every teaching story uses non-identity Wq. Fixed the animation's left panel to display actual input points explicitly, rather than assuming its first state is the input (an assumption previously hidden by Wq=I in the ReLU story).
- Replaced the remaining “weights snapped” storyboard label with the actual rounded-weight map operation. Core algorithms are unchanged.
- Targeted suite: **74 passed**. Visually inspected revised ReLU figures. All 18 display equations and 52 inline expressions pass strict VS Code KaTeX rendering. Local browser verification confirms actual-input data, nonidentity movement, ReLU collision, final correction and zero JavaScript errors.

### 2026-10-11 — derive R/U from the geometric repair problem

- Rebuilt Part 2 around the available repair channel: a linear motion of the quantized preactivation must move merged samples together; the remainder requires input distinctions absent from that channel. Figure 2's A/B midpoint and separate xU displacements motivate R and U before matrix subtraction appears.
- Formalized “best” as the least-squares map objective with equally weighted input directions, then defined U as the still-required displacement map. Retained the identity-centered choice among minimizing repairs and the explicit closed-form solution.
- Revised Figure 2 to show one shape in each of its first three panels. The final panel overlays the repaired line and draws forward xU displacements from the shared point to the FP targets; removed arrows suggesting the collapsed representation alone could recover those targets.
- Moved the numerical-rank qualification into a collapsed note at the end of the research document. Kept bias preservation in the main derivation because it defines the affine repair used by the ledger.
- Strengthened the rank-loss geometric test with independent midpoint and xU coordinates, and a least-squares Pythagorean check that another linear repair cannot improve the map error. Targeted suite: **74 passed**. Visually inspected Figure 2; all 20 display equations and 39 inline expressions render in strict VS Code KaTeX; `git diff --check` passes.

### 2026-10-11 — scalar measurement geometry before matrix R/U

- Rewrote Section 2 in the requested order: quantization changes a weight arrow's length and direction; scalar output-side gain reaches only its parallel projection; rotated fibers make the unrecoverable input distinction visible; matrix R/U generalizes the same measurement-channel problem; exact rank collapse appears later as Figure 2b.
- Added a quantized single-neuron fixture using the existing quantizer/decomposition: w=(1,0.55), wq=(1,1), r=0.775, u=(0.225,-0.225). Both maps have rank 1. A/B share Q response 0.6, but need FP responses 0.69/0.24 and missing additions +0.225/−0.225.
- Added reproducible Figure 2 showing w/wq/rwq/u in input-direction space, orthogonality, and intersecting Q/FP level sets with named A/B. Exact endpoints and responses are saved in `scalar_measurement_fixture.json`. Reproduction: `MPLCONFIGDIR=/tmp/qgeom-mpl XDG_CACHE_HOME=/tmp/qgeom-cache .venv/bin/python -m experiments.quant_geometry_scalar_measurement`.
- Derived the matrix formulas after the geometric problem by writing R=I+C and taking the minimum-norm least-squares increment. Kept numerical qualifications collapsed at the end and corrected references to the later rank-collapse figure.
- Added two independently specified regression tests for the scalar projection, unchanged rank, changed direction/magnitude, equal-Q/distinct-FP fibers, signed missing terms, and least-squares optimality. Targeted suite: **76 passed**. Visually reviewed the new figure; all 25 display equations and 56 inline expressions pass strict VS Code KaTeX rendering. Core quantization/ledger/oracle algorithms are unchanged.

### 2026-10-11 — exact inverse versus architectural dimension reduction

- Added the missing Section 2 bridge: square invertible Wq has R=Wq^{-1}W and U=0 before activation. Explained why any changed map is exactly repairable through this representation and why rectangular full-row-rank maps can also retain every input direction.
- Added a CNN patch example (576 input coordinates to 64 channel measurements for a 3×3, 64-input/64-output convolution) grounded in the official convolution weight shape. Defined the repair family as mixing channels at one output location; spatial correction uses a larger channel. This is conceptual motivation, not a new CNN adapter.
- Added Figure 2b with a bounded-2-bit 3→2 fixture whose FP and Q maps both have rank 2. Showed different measured input planes, a Q fiber through A/B, a shared output-side repair and distinct FP targets restored by xU. Q singular values sqrt(3) and 1 rule out a tiny discarded singular direction in this fixture. The earlier exact quantization rank-drop example is now Figure 2c.
- Added reusable fixture construction, `experiments/quant_geometry_rectangular_measurement.py`, and an exact JSON record of matrices, ranks, singular values and sample states. Reproduction: `MPLCONFIGDIR=/tmp/qgeom-mpl XDG_CACHE_HOME=/tmp/qgeom-cache .venv/bin/python -m experiments.quant_geometry_rectangular_measurement`.
- Added one independent rectangular regression and strengthened the square-map test with an inverse-system check. Targeted suite: **77 passed**. Visually inspected the new figure; all 28 display equations and 67 inline expressions render under strict VS Code KaTeX. No core algorithm changed.

### 2026-10-11 — restore the map/repair/activation conceptual spine

- Replaced Section 2's accumulated inverse-problem tutorial with the general map-repair question from Figure 1. Introduced the geometric least-squares objective, immediately distinguished it from calibration-conditioned reconstruction, established ReLU noncommutation and its two-term ledger, and only then introduced U to complete the three-term ledger.
- Reorganized the research note into a main argument through the exact ledger and propagation/oracle, followed by “Understanding the remainder U” and “Technical details.” Relocated the scalar/fiber example, 3→2 and CNN example, and exact rank-collapse example as Figures A1–A3. Moved pseudoinverse construction, rank/fiber criterion and proof, bias handling, correction nonuniqueness, full Gram accounting, and example matrices out of the narrative spine. Numerical policy remains a collapsed technical note.
- Preserved formal definitions and exact state-comparison figures for every component. The preactivation ReLU obstruction is now Figure 3; component comparisons and cancellation are Figures 4–7. Updated the reading guide and experiment index to match.
- Added research Figure 2 and `experiments/quant_geometry_reconstruction_comparison.py` using Figure 1's nonidentity quantized map. With 859 fixed grid/circle calibration samples (A/B excluded), R has effectively zero map discrepancy; fitted C has squared Frobenius map discrepancy `0.00470510`, while reducing calibration activation MSE from `0.00517103` to `0.00475163`. Normal-equation residual: `7.84e-14`. Exact matrices, calibration definition, precision and objectives are in `repair_vs_reconstruction_fixture.json`; reproduction and contribution mapping are in their separate documents.
- Visually inspected the new comparison figure; its script verifies distinct matrices and the reversed preference of map versus activation objectives. Targeted suite: **77 passed**. All **31 display equations and 71 inline expressions** render with strict VS Code KaTeX. Local links/assets and `git diff --check` pass. Core algorithms and milestone statuses are unchanged; this illustration does not complete the held-out correction-family milestone M5.

### 2026-10-11 — enforce prerequisite order in the main argument

- Removed activated-output notation from the opening linear-map discussion. The initial calibration distinction now compares map repair with data-weighted linear response fitting only.
- Introduced ReLU as coordinate clipping, defined its action and the FP/Q outputs, then showed repair-versus-clipping order and the exact two-term account. Only afterward introduced the post-ReLU calibration objective and comparison. The clipping storyboard is Figure 2; the reconstruction comparison is Figure 3. Updated their references and contribution records.
- Read through the main argument for definitions preceding use: x/W/Wq → R → ReLU and zq/h/y → activated fit C → U → ledger states/components → propagation and oracle. Kept the zero-bias convention through propagation and defined arriving Q output hq before the oracle; the float-bias extension remains in the technical appendix.
- Existing figures and algorithms are unchanged. The new order, internal references, local links/assets, and `git diff --check` pass; **32 display equations and 72 inline expressions** render under strict VS Code KaTeX. The existing targeted acceptance result remains **77 passing tests**; no implementation change required a repeated test run.

### 2026-10-11 — keep the inverse guarantee and correction precision in Part 2

- Restored the square-invertible result beside the main repair objective: mathematically R=Wq^{-1}W repairs every input before activation. Figure 1 explicitly illustrates this case with 2-bit Wq and full-precision R. Removed the duplicate inverse derivation from the technical appendix; retained its broader rectangular/rank/fiber criterion there.
- Stated in Part 2 that many ANN layer maps are rectangular (width changes and patch-to-channel convolution), so square-invertible exact repair is not a general network assumption. Preserved the distinction that widening maps can retain all input directions and narrowing maps can lose FP-relevant distinctions without a rank drop. U=0 is stated after U is defined, preserving prerequisite order.
- Made the correction domain explicit: R is an unrestricted real matrix, implemented at W's full floating-point precision and never quantized by this apparatus. Documented Wq's dequantized floating storage separately from its low-bit coefficient grid. Quantizing R would impose an additional correction constraint.
- Checked Figure 1 independently in float32/float64: matching R/W dtypes, inverse-system agreement, and R coefficients off Wq's grid; reconstruction residuals `2.98e-08` / `1.11e-16`. Geometry-story tests: **7 passed**. All **32 display equations and 72 inline expressions** render under strict VS Code KaTeX; local links/assets, prerequisite ordering, and `git diff --check` pass. Core algorithms, figures and milestone statuses are unchanged.

### 2026-10-11 — derive the remainder's role in the ledger

- Replaced “Some map deformation can remain before activation” with “From exact repair to the general three-term ledger.” Explained its position: exact-repair examples first isolate the ReLU obstruction; relaxing exact linear repair then requires another comparison to account for the actual FP output.
- Motivated U by the geometric endpoint gap: the existing two terms reach the activated R-repaired shape, which need not be the FP shape. Defined the preactivation displacement xU, then added its activated effect explicitly to obtain the general three-term identity. The next section names and draws those terms; Figure 6 supplies the third comparison.
- Stated the broader role: distinguish incomplete linear-map repair from activation obstructing that repair. Both can occur together. Evaluating xU uses the original input for the analysis; it is not another operation in the output-side repair channel. Retained U=0 for square invertible Wq and all formal component definitions.
- Core algorithms and figures are unchanged. Checked the exact-to-general identities on the existing three story fixtures, local links/assets and `git diff --check`; **33 display equations and 73 inline expressions** render under strict VS Code KaTeX. No repeated acceptance-suite run was needed for this explanation change.

### 2026-10-12 — compare map, preactivation data, and activation data repair

- Formalized all three objectives with explicit names: R_map, R_pre,data, and R_post,data. Introduced the preactivation data objective before ReLU and the activation data objective after ReLU; placed all three objectives together in a comparison table. The ledger still uses full-precision R=R_map. Preserved the square-invertible guarantee and the architectural rectangular-map distinction.
- Replaced Figure 3's square-invertible fixture, which can hide the first two objectives' difference, with the existing full-rank 3→2 W/Wq. A deliberately correlated input cloud with finite thickness gives rank-3 calibration data, while both maps retain rank 2. The new four-column figure compares uncorrected Q and all three fits in two rows: preactivation and activation-then-correction. Fixed scales, FP reference shapes and named A/B correspondences show the actual motions; no further ReLU is applied after correction.
- Added Figure 1b showing equal directional weighting on a sphere beside the calibration cloud near a tilted input plane. Calibration includes three parallel sheets (2,577 equally weighted samples); Figure 3 displays the central sheet and reports losses on the entire cloud. A/B are excluded on every sheet. The preactivation data fit matches the central sheet exactly, with its blue shape covering the dashed FP target; finite thickness explains its nonzero calibration loss.
- Each repair wins its own objective: squared Frobenius map loss `0.084167` for R_map; calibration preactivation MSE `0.001894` for R_pre,data; calibration post-ReLU MSE `0.010080` for R_post,data. Normal-equation residuals are at most `1.38e-12`. Verified the analytic preactivation fit, central-sheet agreement, the second-moment trace identity for every repair, and agreement between map and isotropically weighted preactivation fits.
- Updated `experiments/quant_geometry_reconstruction_comparison.py`, exact fixture data, implementation reproduction, reading guide, experiment index and contribution mapping. Added the input second-moment derivation in the technical appendix. Updated the U transition to refer to Figure 3's incomplete map repair, preserving the role of the third ledger comparison.
- Visually inspected both figures, including a camera correction that makes the tilted plane visible rather than edge-on. Reproduction assertions pass; targeted acceptance suite: **77 passed**. All **36 display equations and 87 inline expressions** render with strict VS Code KaTeX; local links/assets, prerequisite order and `git diff --check` pass. This remains a controlled explanatory fixture; no held-out correction-family milestone or core algorithm change is claimed.

### 2026-10-12 — remove the detached Figure 1b input-space detour

- Removed Figure 1b from the research note, its generated asset, and its plotting helper. The sphere/input-cloud comparison switched geometric objects without showing the consequence for repair; the narrative now explains that consequence through Figure 3's different motions of the same Q points.
- Preserved all three formal objectives and their comparison table. The input second-moment derivation and exact correlated-data construction remain in the technical appendix. Simplified Figure 3's caption/footer to distinguish displayed representative inputs from the full calibration set without requiring a new input-space picture.
- Updated reproduction and contribution records so the removed asset is not regenerated or referenced. Regenerated and visually inspected Figure 3; all comparison assertions still pass and numerical results are unchanged. Local links/assets, prerequisite order, strict VS Code KaTeX (**36 display equations / 87 inline expressions**) and `git diff --check` pass. No core algorithm or acceptance-suite change required another broad test run.

### 2026-10-12 — name the activated U effect from the geometric story

- Replaced “range error” with “residual deformation error,” written e_U, throughout the research note: ledger record, formal component definition, telescope, bias extension, cancellation example and Gram component labels. The name now follows the already-defined residual map deformation U and distinguishes its preactivation displacement from its activated effect.
- Updated Figure 6, the cancellation legend and diagnostic titles/legends, including the U matrix label. Regenerated the main ledger figures and diagnostic/MLP figures. Visually inspected the residual state-pair, cancellation, component diagnostic and Gram/accounting figures for label readability.
- Retained the historical Python field/dictionary key `range` and artifact filenames. Documented their mapping to residual deformation error e_U in the implementation README and class docstring; no compatibility note was added to the conceptual narrative. Updated current protocol and contribution terminology.
- Compared regenerated fixture/ledger/oracle manifest sections with the prior manifest: numerical results are identical. Targeted acceptance suite: **77 passed**. Local links/assets, narrative prerequisites, `git diff --check`, and strict VS Code KaTeX (**36 display equations / 87 inline expressions**) pass. Component definitions, signs, ordering and arithmetic are unchanged.

### 2026-10-12 — reconcile the ledger in the forward repair direction

- Rebuilt Sections 3–5 around the hierarchy: observed correction c_total=y−h → available map motion c_R=hR−h → outstanding balance y−hR → activation-order balance c_NL plus incomplete-map balance c_U. The main text first treats U=0, then derives the general boxed three-contribution identity. Removed the leading four-state table and the peer-error taxonomy.
- Preserved Section 2's map/repair/calibration/ReLU/U progression. Moved its duplicate error-accounting algebra into the new ledger, shortening Section 2 from 104 to 92 lines. Sections 3–5 are reduced from 771 to 641 words despite adding the network visual. Formal definitions and exactness are retained; state indices do not lead the explanation.
- Reversed Figures 4–7 to the forward path h→hR→phi(zqR)→y. Figure 4 now shows the total owed, R's motion, and its remaining balance in that order. Figures 5/6 show the two reasons for that balance; Figure 7 shows c_R being undone by c_NL before c_U reaches FP. Saved actual endpoints, contributions and balances in `repair_ledger_figure_fixture.json`; the script checks them against negatives of the existing ErrorLedger fields and independent expected coordinates.
- Added Figure 8 for incoming/local balances and the oracle as a direct reset, with local (−0.175,−0.45), incoming (−0.405,+0.045) and total (−0.58,−0.405). Verified the network decomposition and sign mapping on every layer of biased float64 MLP runs, including a delayed oracle reset. Also checked the affine-bias forward identity on all three geometry fixtures.
- Kept the stored Q-minus-FP API intact. Technical Details now contain the bias extension, explicit field-to-correction sign mapping, correction-extension dependence, and Gram machinery in the forward convention. Gram values are unchanged by reversing every sign. Updated reading-guide links, reproduction instructions, class documentation and contribution records.
- The account follows sequential operations and their outstanding balances; no additional approximation mechanism was introduced. Core quantization, decomposition, ledger arithmetic and runner behavior are unchanged. Visually inspected all five forward figures. Reproduction assertions, local links/assets, `git diff --check`, strict VS Code KaTeX (**36 display equations / 90 inline expressions**) and the targeted suite (**77 passed**) pass.

### 2026-10-13 — follow one map through deformation, repair, and ReLU

- Figures 1, 2, 4, and 5 now use W=[[1,−0.4],[0.65,0.85]], Wq=[[1,0],[1,1]], R=[[1,−0.4],[−0.35,1.25]], U=0, and the same A=(−0.8,−0.1), B=(−0.6,−0.2). The deformation and ReLU views share their fixture at the source rather than repeating independently chosen matrices.
- Regenerated Figure 1, the dependent Figure 4, exact story/ledger JSON and the populated moving view. Figures 2 and 5 retain their existing geometry. Updated captions and consolidated the duplicate matrix examples in Technical Details. Figure 4 now explains that R moves the overall clipped shape while both marked zero activations stay at zero; their balance is reconciled in Figure 5.
- Preserved the separate network/oracle fixture and its coordinates explicitly, removing its accidental dependency on the deformation fixture. Existing conceptual edits in the foundations document were retained.
- Updated independent expected coordinates and extended the existing geometric test to verify shared input identities and preactivation endpoints. Both artifact generators, animation-payload correspondence, the targeted suite (**77 passed**), strict VS Code KaTeX (**35 display equations / 90 inline expressions**) and `git diff --check` pass. Visually inspected Figures 1, 2, 4, and 5. Float32/float64 inverse checks pass, with maximum map residuals 1.19e-07 / 1.11e-16; verification and reproduction documentation are updated.
