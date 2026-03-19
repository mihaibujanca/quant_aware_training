# Cycle 050 Audit — Oracle-guided adaptive compression ratio (GPT-2 FFN)

```
status: PASS
original_outcome: REFUTED
audited_outcome: REFUTED (confirmed)
guardrail_flags: G4-counterintuitive
reason: >
  The counterintuitive result (oracle worse than uniform) is genuine and well-explained.
  Pre-correction PPL at k=4 is nearly identical (oracle ~10.46% vs uniform ~10.44%),
  confirming the P1-guided variable ratios don't help block-level damage. The critical
  divergence is at the correction step: lstsq MSE *increases* for oracle (1.71->1.90,
  all 3 seeds) while *decreasing* for uniform (1.71->1.37). Variable hidden dims per
  block produce structurally incompatible residuals for a single affine correction,
  which is a genuine interaction effect, not a bug. The refutation is of the hypothesis
  that P1-guided allocation improves the full pipeline, not of P1 as a damage predictor.
recommendations:
  - The finding implies that any adaptive-ratio scheme must co-design the correction
    step (e.g., per-block correction or ratio-aware correction fitting).
  - Test whether per-block corrections (instead of a single affine at block 8) would
    rescue the oracle allocation.
  - This closes the "variable ratio" direction unless correction compatibility is solved.
```

## G1: Implementation correctness

**PASS.** Code is clean and correct:
- `per_layer_hidden` correctly overrides the default `hidden_fraction` in `_layer_target_hidden()`.
- `TARGET_BLOCKS = [7, 5, 4, 1]` accumulation order is identical for both oracle and uniform -- fair comparison.
- Correction fitted on `probe_train`, evaluated via `_evaluate_model` on `eval_val` -- proper train/test split.
- Hook placement at `CORRECTION_LAYER = 8` (one layer after the last compressed block) is correct.
- The `fit_correction` lstsq formulation (affine: `[h_c | 1] -> h_f`) is standard.

## G2: Reproducibility

**PASS.** 3 seeds (42, 123, 7), 200 eval sequences each. Key effects are consistent across all seeds:
- Oracle MSE worsens in all 3 seeds: 1.71->1.90, 1.71->1.88, 1.74->1.91
- Uniform MSE improves in all 3 seeds: 1.71->1.39, 1.71->1.36, 1.74->1.41
- Oracle correction recovery is negative in all 3 seeds: -2.23pp, -2.94pp, -0.44pp
- Uniform correction recovery is positive in all 3 seeds: +3.11pp, +3.60pp, +5.52pp
- Standard deviations are small (0.82% oracle, 0.69% uniform).

## G3: Results interpretation

**PASS.** The interpretation correctly identifies the interaction effect:
1. At k=4 uncorrected, oracle (~10.46%) and uniform (~10.44%) are essentially tied -- the variable ratios don't help at the raw compression level.
2. The correction step is where the divergence occurs: uniform benefits from correction (-4.1pp mean), oracle is *hurt* by correction (+1.9pp mean).
3. The manifest correctly attributes this to "incompatible structure for a single affine boundary correction" rather than claiming P1 is a bad predictor.
4. Comparison against reference baselines (c042 10x: +8.45%, c039 6x: +8.35%) is appropriate.

## G4: Counterintuitive results

**INVESTIGATED -- EXPLAINED.** Oracle being worse than uniform is surprising but the mechanism is clear:

The variable hidden dimensions (307, 312, 342, 389) create heterogeneous residual structure at block 8, where the single affine correction is fitted. The lstsq system tries to fit one linear map to undo damage from blocks with different compression geometries. With uniform compression (all blocks at hidden=335), the residual structure is homogeneous and the affine map can capture it. This is analogous to how a single linear correction cannot undo nonlinear distortion -- the mixing of different compression ratios introduces structure that a single affine map cannot model.

The MSE evidence is unambiguous: correction makes things worse for oracle (1.71->1.90) and better for uniform (1.71->1.37). This holds across all 3 seeds with zero exceptions.

## G5: Method failure vs genuine refutation

**Genuine refutation of the hypothesis as stated.** The hypothesis was that P1-guided variable ratios would improve the end-to-end pipeline (compression + correction). The experiment correctly tests this. The refutation does *not* invalidate P1 as a local damage predictor -- it shows that optimizing per-block ratios independently is counterproductive when the correction step assumes structural homogeneity. This is a useful negative result that constrains future work.
