# Geometry-Guided Quantization Correction: Overnight 12h Experiment Report

## Abstract

This report summarizes an 18-setting overnight experiment on transformer quantization correction.
We tested two hypotheses:
1. whether pre-activation quantization error is linearly recoverable at 4-bit correction precision (Q1), and
2. whether quantization-time statistics provide transferable correction signal beyond layer identity (Q2).

Main outcome: Q1 is not supported in the current 4-bit correction regime, while Q2 is supported in 17/18 settings.
In practice, feature-conditioned correction is consistently better than layer-index-only correction on held-out correction MSE, but deep/wide models show occasional downstream instability.

## 1. Objective

The scientific objective is to identify which geometric error components remain recoverable after quantization and to determine whether correction can be driven by measurable layer statistics rather than per-layer bespoke tuning.

## 2. Run Metadata

- Run directory: `runs/geometry_overnight_12h`
- Completion: successful (`runs/geometry_overnight_12h/run.log`)
- Settings: 18 total (`3 seeds x 6 architectures`)
- Seeds: `42, 123, 999`
- Architectures: `(d_model, n_layers) in {(64,2), (64,4), (128,4), (128,6), (256,4), (256,6)}`
- Aggregated summary: `runs/geometry_overnight_12h/master_summary.csv`

## 3. Terminology (Disambiguation)

### 3.1 Error and Recovery Terms

- Pre-activation error:
  difference between float and quantized pre-activation tensors at the same layer/sub-layer.
- Linear recovery map:
  best-fit linear operator trained to map quantized pre-activation back toward float pre-activation.
- `R_lin_float`:
  fractional error reduction from the best linear recovery map in float precision.
- `R_lin_q4`:
  fractional error reduction from the same recovery concept under 4-bit constrained correction weights.

Interpretation of `R_lin_*`:
- `R_lin > 0`: recovery helps
- `R_lin = 0`: no net effect
- `R_lin < 0`: recovery operator worsens error

### 3.2 “Embedding” vs “Features” in Q2

Three correction-input variants were evaluated:
- `embedding_only`:
  correction model sees only a learned representation of layer identity/position (no runtime quantization statistics).
- `features_only`:
  correction model sees only quantization-time statistics for that layer sample (for example error magnitudes and geometry proxies), without layer-identity embedding.
- `features_plus_embedding`:
  correction model sees both quantization-time statistics and layer-identity embedding.

So, in this report, “embedding” means layer-context token, not token embedding from the language model input pipeline.

## 4. Experimental Design

## 4.1 Q1: Linear Recoverability Before Nonlinear Collapse

Hypothesis Q1:
A substantial fraction of pre-activation quantization error remains linearly recoverable at 4-bit correction precision before nonlinear collapse dominates.

Assessment policy:
- Compute per-layer `R_lin_float` and `R_lin_q4`.
- Compute Spearman correlation between `correctability_score` and `R_lin_q4`.
- Verdict criteria:
  - Supported: `median(R_lin_q4) >= 0.35` and `Spearman >= 0.35`
  - Refuted: `median(R_lin_q4) < 0.24` and `Spearman < 0`
  - Unclear: otherwise

## 4.2 Q2: Transferable Corrector Signal from Quantization-Time Statistics

Hypothesis Q2:
Quantization-time statistics contain correction signal beyond layer identity; models using these statistics should outperform layer-index-only models.

Assessment policy:
- Compare `embedding_only`, `features_only`, `features_plus_embedding`.
- Metrics:
  - held-out correction MSE (elementwise prediction quality)
  - downstream `loss_gain` over quantized baseline
- Verdict criteria:
  - Supported if:
    - `features_only` improves MSE over `embedding_only` by at least 15%, and
    - best feature-based variant improves downstream loss by at least `+0.0100`
  - Refuted if `features_only` is worse and no variant improves downstream loss
  - Unclear otherwise

## 5. Quantization/Correction Regime

- Quantization focus in this run: 4-bit quantized analysis/correction path.
- Q1 compares unconstrained float linear recovery (`R_lin_float`) to 4-bit-constrained recovery (`R_lin_q4`).
- Q2 evaluates compact correction predictors using layer context and/or quantization-time statistics.

## 6. Results

## 6.1 Q1 Outcome

Per-setting verdicts (`runs/geometry_overnight_12h/master_summary.csv`):
- `refuted = 13`
- `unclear = 5`
- `supported = 0`

Global per-layer aggregates (156 layer points):
- `R_lin_float`: mean `0.3368`, median `0.3483`, positive fraction `1.0000`
- `R_lin_q4`: mean `-0.1194`, median `-0.0688`, positive fraction `0.3718`
- fraction with `R_lin_q4 > 0.2`: `0.1346`
- Spearman(`correctability_score`, `R_lin_q4`) = `-0.4481`
- Spearman(`correctability_score`, `R_lin_float`) = `-0.8082`

Depth profile for `R_lin_q4`:
- Early layers: mean `-0.3530`, median `-0.2949`, positive fraction `0.0556`
- Mid layers: mean `0.0307`, median `0.0070`, positive fraction `0.5000`
- Late layers: mean `0.2061`, median `0.2036`, positive fraction `1.0000`

Sub-layer profile for `R_lin_q4`:
- Attention: mean `-0.1507`, median `-0.0754`, positive fraction `0.3590`
- FFN: mean `-0.0881`, median `-0.0564`, positive fraction `0.3846`

Interpretation:
- A linear correction exists in float space.
- Imposing 4-bit correction constraints largely removes that recoverability, especially early in the network.
- Current `correctability_score` is not aligned with observed 4-bit recoverability in this regime.

## 6.2 Q2 Outcome

Per-setting verdicts (`runs/geometry_overnight_12h/master_summary.csv`):
- `supported = 17`
- `unclear = 1`
- `refuted = 0`

Variant-level aggregate findings:
- Best held-out MSE variant by setting: `features_plus_embedding` in `18/18` settings.
- `features_only` vs `embedding_only` held-out MSE improvement:
  - mean `+30.38%`, median `+29.23%`, min `+22.34%`, max `+42.50%`
- Mean downstream `loss_gain`:
  - `embedding_only`: `0.6546`
  - `features_only`: `0.7504`
  - `features_plus_embedding`: `0.7575`

Observed instability mode:
- In two `(d_model=256, n_layers=6)` settings, `features_only` had negative downstream gain while `embedding_only` remained positive.
- `features_plus_embedding` mitigated one of these two failures but stayed slightly negative in one seed.

Interpretation:
- Quantization-time statistics provide strong transferable signal for correction.
- Combining runtime statistics with layer context is the most reliable approach in this sweep.
- Downstream objective alignment is still imperfect in deep/wide settings.

## 6.3 Architecture-Level Verdict Summary

- `(64,2)`: Q1 unclear (3/3), Q2 supported (3/3)
- `(64,4)`: Q1 refuted (2/3), unclear (1/3); Q2 supported (3/3)
- `(128,4)`: Q1 refuted (2/3), unclear (1/3); Q2 supported (3/3)
- `(128,6)`: Q1 refuted (3/3); Q2 supported (3/3)
- `(256,4)`: Q1 refuted (3/3); Q2 supported (3/3)
- `(256,6)`: Q1 refuted (3/3); Q2 supported (2/3), unclear (1/3)

## 7. Scientific Conclusion

- Q1 conclusion:
  the proposition “linear recoverability remains substantial at 4-bit correction precision” is not supported by this run.
- Q2 conclusion:
  quantization-time statistics do carry transferable correction signal beyond layer identity, and should remain central in the next iteration.

## 8. Limitations

- This report is specific to the tested architectures, seeds, and 4-bit correction regime.
- Q1 evidence indicates a precision bottleneck but does not isolate whether the bottleneck is quantizer design, correction parameterization, or optimization dynamics.
- Q2 uses MSE and downstream loss as evaluation; causal interpretation of which feature families drive gain is not yet isolated.

## 9. Next Experimental Priorities

1. Keep `features_plus_embedding` as the default Q2 baseline.
2. For Q1, evaluate whether improving correction precision/parameterization restores positive `R_lin` in early layers.
3. Add analyses that separate correction-MSE improvement from downstream-loss improvement in deep/wide settings.

## 10. Reproducibility Pointers

- Hypothesis card: `runs/geometry_overnight_12h/hypothesis_card.md`
- Aggregate results: `runs/geometry_overnight_12h/master_summary.csv`
- Per-setting summaries: `runs/geometry_overnight_12h/seed*/assessment_summary.md`
- Raw Q1 metrics: `runs/geometry_overnight_12h/seed*/q1_layer_metrics.csv`
- Raw Q2 metrics: `runs/geometry_overnight_12h/seed*/q2_variant_metrics.csv`
