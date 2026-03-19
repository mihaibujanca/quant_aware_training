# Cycle 063: Item 79 — Per-Block Boundary Correction (Stacking-Aware)

## Task
Revisit multi-point correction with stacking-aware fitting on GPT-2 FFN compression. After compressing k=4 blocks at 6× with topk+lstsq, fit one affine correction per block **in cascade order**: C_1 on float inputs, C_2 on (compressed_1+C_1) outputs, etc. This separates correction fitting from stacking interference (issue from cycle 043).

## Hypothesis
Cascaded boundary corrections can reduce stacked k=4@6× PPL degradation to ≤+12% (vs +30.2% uncorrected), by training each correction to fix compounding error rather than isolated block error.

## Method
- Compress 4 FFN blocks at 6× compression ratio (topk+lstsq)
- Fit affine corrections C_i = A_i·h + b_i at each block output
- Use cascaded re-probing: each correction fitted on previous block's corrected output
- Baseline: cycle 061 result (+4.9% PPL with gradient polish, +30.2% without)

## Success Criteria
- **Confirmed**: cascaded corrections achieve ≤+12% PPL degradation
- **Refuted**: corrections overfit or degrade PPL beyond uncorrected stacking
- **Inconclusive**: corrections improve (+15–25%) but underperform gradient polish alone (+18.4%)

## Dependency
None — ready to run.
