# Quantization geometry progress

Plan: [quantization_geometry_plan.md](quantization_geometry_plan.md)

Started: 2026-10-11. Branch: `codex/quant-geometry-framework`.
Baseline: `0be49df7` (`Snapshot research state before quantization return`).

## Tracking rules

- Status: TODO, IN PROGRESS, PROTOTYPE, DONE, or BLOCKED.
- PROTOTYPE means related code exists but does not meet the plan's acceptance criteria.
- Mark DONE only with reusable implementation, relevant passing tests, and required experiment artifacts. Record paths and verification commands below.
- For each experiment record revision, seed, dataset/split, model dimensions, quantization scheme/scales/bits, selected quantized layers, correction sites, tolerances, and output paths.
- Update this file with each meaningful implementation change. Keep research findings separate from implementation completion; record decisions below. Step-by-step history lives in git.
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

## Milestones and work queue

| ID | Status | Deliverable / completion gate |
|---|---|---|
| A0 | DONE | Read plan, audit relevant reusable code and prototypes, create branch and this tracker. |
| M1 | DONE | `weight_quantizer.py`, `map_decomposition.py`, `ledger.py`; exactness/orthogonality tests, bounded bit grid, bias-aware per-sample vectors. |
| M2 | DONE | `instrumented_mlp.py` reuses `ForwardTrace`, selects arbitrary weight-quantized layers, records FP/raw pre/post and corrected consumed post states plus metadata; local/propagated identity tested. |
| M3 | DONE | Shared `ResidualOracle` and `CorrectionBridge`; immediate intervention tested independently at every layer of a variable-width MLP; raw damage preserved and FP downstream trajectory restored. |
| M4 | DONE | All-layer immediate mode tested: inputs and consumed outputs match FP at every layer while local damage remains visible. This is apparatus validation on an untrained MLP, not a trained-model research result. |
| M5 | PROTOTYPE | Existing selected-site oracle and SVD experiments are ingredients only. Isolate one source layer, delay injection, fit linear/affine corrections on calibration data, evaluate held-out residuals/ranks/amplification versus depth. |
| L1 | TODO | Ordered component counterfactual propagation with explicit ordering and exact downstream telescoping. |
| L2 | PROTOTYPE | Local `ErrorLedger.gram()` exposes interaction; downstream recoverability curves and tolerance-based horizon still TODO. |

The first three requested next steps are complete: reusable quantizer/decomposition/ledger, exact tests and single-layer figures, and shared MLP tracing/oracle extraction. Core code stays under the installed `aleph/qgeom` package.

## Next steps (planned 2026-10-13)

The research note's single-layer argument is in a state we are happy with. Next we test its claims on a trained network.

1. **Ledger on a trained network.** Train a 2D classification MLP (spirals) and confirm FP accuracy first. Quantize weights only and compute per-layer $c_R$, $c_{\rm NL}$, $c_U$ and their Gram matrix, under ordinary quantized inference and under the immediate oracle. Include both square and narrowing layers. The note predicts $c_U\approx0$ when $W_q$ stays invertible (square or widening, full rank) and $c_U\ne0$ for narrowing layers. Check this, and check how $c_{\rm NL}$ compares in size.
2. **Calibration fitting versus map repair (M5, and Section 9 of the note on real data).** Fit $R_{\rm pre,data}$ and $R_{\rm post,data}$ on a calibration split and evaluate on held-out data. For each correction, report map error $\lVert W-W_qR\rVert_F$, held-out output error, and which ledger terms it reduces. The toy example's claim to test: the data fit lowers calibration error while moving $W_qR$ further from $W$, and its gains do not transfer when input correlations change.
3. **Ordered component propagation (L1)** only after 1–2, before interpreting where each term goes downstream. MNIST or transformers only after these protocols are validated.

Small documentation leftovers: Figure 9's column label says "fit activations" where the note says "output fit"; Section 5 still uses "projection" and singular values without explanation; `quantization_geometry_plan.md` uses old term names (uncompensated/range) and its module-design sections describe code that now exists.

## Mathematical / protocol decisions

- Implemented row-vector weights `[in, out]` internally; the MLP adapter transposes `nn.Linear.weight` at the boundary.
- Resolved float bias with the affine repair `T(v)=vR+b-bR`. The forward account uses c_R=T(h)-h, c_NL=phi(T(zq))-T(h), and c_U=y-phi(T(zq)). Their negatives are the stored implementation fields. This preserves exactness and zero activation-order balance for identity activation.
- The research ledger reconciles FP minus Q, the correction still owed. The implementation retains Q-minus-FP fields; their negative mapping is in Technical Details. The oracle adds the forward total correction.
- Full row rank of a widening `Wq` gives `U=0` in exact arithmetic if no nonzero singular values are discarded. Under truncation, `U` is a correction-policy remainder and need not be orthogonal to the full `Wq` or prove lost information. See the numerical note in the research note's appendix B.
- A same-input ledger measures local creation. Comparing layers fed different upstream states also includes propagation and needs separate accounting.
- Components are not generally orthogonal. Their norm/energy ratios are not additive percentages of total error; report cross terms or use clearly defined allocations.
- Attribution also depends on the declared extension of `R`: a change `R -> R+K` with `Wq K=0` can alter post-ReLU components while preserving the original linear map and total error. The implementation fixes the identity-centered convention.
- An exact additive oracle can restore FP states after ReLU. The activation-order balance compares the chosen map repair on opposite sides of ReLU; a nonzero value does not prove that every other correction family fails.
- Recoverability horizons require a named correction family, held-out evaluation and tolerance. Error may become easier to recover again at later bottlenecks; report full curves alongside any first-threshold-crossing statistic.

## History

The dated log of earlier implementation and documentation steps was removed on 2026-10-13; see `git log` on this branch.
