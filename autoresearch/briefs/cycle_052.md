# Cycle 052 Handoff Brief

## Task
Item 68: **Corrected lstsq refit residual rank analysis — GPT-2 FFN**

Retry of cycle 051 (design-review-rejected) with corrected discarded-subspace projection.

## Context
- **Blocker**: Cycle 044 (domain robustness) found Shakespeare→WikiText-2 zero-shot degradation +120.5% PPL, signaling corpus specificity
- **Hypothesis**: Corpus gap is caused by *discarded-subspace* information loss (lost when channels are removed), not lstsq coefficient sensitivity
- **Method**: For blocks [7,5,4,1] at 6×:
  1. Decompose refit residual E = Y_float - Y_compressed into: (a) in-column-space(A_kept), (b) orthogonal
  2. Compute rank-95 of orthogonal component separately on Shakespeare vs WikiText-2 probes
  3. If WikiText rank >> Shakespeare rank, discarded subspace carries corpus-dependent info

## Key Findings from Predecessors
- **Cycle 044**: Shakespeare probed +6.3% (great), WikiText-2 probed +31.9% (bad), Shakespeare→WikiText zero-shot +120.5% (catastrophic)
- **Cycle 039**: Residual correction confirmed (k=4 +8.3%, k=5 +12.5%), high-rank (rank-95=392/768)
- **Cycle 045**: Recipe validated on Shakespeare but requires corpus-specific tuning for WikiText-2

## Criterion
- **Confirmed**: Orthogonal residual rank-95 on WikiText ≥2× rank-95 on Shakespeare for ≥3/4 blocks → corpus gap is discarded-subspace loss
- **Refuted**: Rank ratio <1.3 → corpus gap is coefficient sensitivity, not subspace loss
- **Inconclusive**: Rank differs but orthogonal component <20% total residual energy

## Dependencies
None. Can run immediately.

## Technical Notes
- Use `torch.linalg.svd` on E; project via `A_kept @ torch.linalg.lstsq(A_kept, E).solution`
- Orthogonal = E minus projection
- Metric: rank-95 threshold (cumulative variance ≥0.95)
- Compare same blocks/ratios/probe sizes as cycle 044 for direct comparability
