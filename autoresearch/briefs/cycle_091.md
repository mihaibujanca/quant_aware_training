# Cycle 091: MI Bottleneck Channel Selection

## Context
- Item 86, P2 moonshot
- Status.md: cycles 071/077 showed **basis irrelevant** (SVD = random)
- This experiment tests whether **channel selection method matters**

## Question
Current compression uses topk (channels by activation magnitude × output weight norm).
Does information-theoretic optimality (mutual information) find better channels?

## Hypothesis
If topk is near-optimal proxy, MI-greedy will select ≥90% same channels → refute.
If MI captures functional importance missed by magnitude, PPL improves ≥5% → confirm.

## Method
1. Blocks [1, 4, 7, 11] at 6× compression (k=128 channels)
2. MI-greedy: forward selection minimizing det(Σ_residual) via MINE or Gaussian approx
3. Compare: MI-greedy + lstsq vs topk + lstsq on single-block PPL

## Win Condition
MI-greedy ≥5% rel PPL improvement in ≥2/4 blocks.

## Dependencies
None. Orthogonal to basis irrelevance (cycles 071/077).

---
**Status**: Ready for execution
**Est. effort**: Medium (MI estimator implementation)
