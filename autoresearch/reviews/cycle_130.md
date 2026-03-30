# Design Review — Cycle 130 (Item 87: Cross-block Procrustes Alignment)

verdict: REVISE
round: 1
issues:
  1. Procrustes residual formula is transposed incorrectly (R4, implementation bug)
  2. Single seed without fallback for ambiguous-range residuals (R3, minor)
feedback: see below

## R1: Premise validity — PASS

Hypothesis is plausible for a MOONSHOT. Cycle 077 showed output manifold isotropy (random basis = SVD after lstsq), and cycle 029 found shared spectral structure across blocks. These don't prove W_down similarity, but they provide weak motivation. Appropriate for a P2 moonshot with clear go/no-go gates.

## R2: Measurement soundness — PASS

Null baseline (independent random Gaussian matrices, shape 768×512): expected Procrustes residual after optimal R is approximately 0.6 (random high-dimensional matrices are nearly orthogonal). The 10% confirmation threshold is well below this, so a low residual would be a genuine signal. The 30% refutation threshold is also well below random, so "all pairs > 30%" would still mean meaningfully non-random alignment — this is fine since the question is practical (can we share weights?), not statistical.

Self-distance sanity check is included. Good.

## R3: Capacity and statistical adequacy — MINOR ISSUE

Procrustes is deterministic given the weight matrices, but the weight matrices depend on topk+lstsq+500 Adam steps with seed 42. For a moonshot go/no-go, 1 seed is defensible IF the residuals are far from the thresholds. **The plan should specify**: if any pairwise residual falls in [5%, 15%] (near confirmation) or [25%, 35%] (near refutation), re-run compression with seeds {42, 123, 7} and report mean ± std of residuals.

## R4: Method-architecture alignment — BUG FOUND

**Step 3 has a transposition error in the residual formula.** The plan passes `(W_i.T, W_j.T)` to `scipy.linalg.orthogonal_procrustes`, which minimizes `||W_i.T - W_j.T @ R||_F` and returns R of shape (768, 768). The residual should then be computed as:

```
||W_i.T - W_j.T @ R||_F / ||W_i||_F
```

or equivalently in original (non-transposed) space:

```
||W_i - R.T @ W_j||_F / ||W_i||_F
```

The plan writes `||W_i - W_j @ R||_F / ||W_i||_F` — this is wrong. `W_j` is (768, 512), `R` is (768, 768), so `W_j @ R` has shape (768, 768), which cannot be compared to `W_i` of shape (768, 512). This would either crash (shape mismatch) or silently produce nonsense if shapes happen to broadcast.

**Step 5 has the same issue**: "replace c_proj_weight with `R @ W_{i*}`" should be `R.T @ W_{i*}` (since R aligns the transposed matrices, the original-space rotation is R.T).

**Fix**: Either (a) compute Procrustes in original space `orthogonal_procrustes(W_i, W_j)` → R of shape (512,512), residual `||W_i - W_j @ R||_F / ||W_i||_F`, replacement `W_{i*} @ R_{ij}^T`; or (b) keep the transposed convention but fix all downstream formulas to use R.T consistently.

Option (a) is cleaner: Procrustes on the (768, 512) matrices directly gives R ∈ R^{512×512} (column-space rotation), which is also cheaper to store and apply.

## R5: Known pitfall scan — PASS

No pitfalls apply. MP1: thresholds well below random null. MP3: Procrustes is closed-form. MP7: weight-space comparison, not post-LayerNorm activations.

## Summary

Two issues, one blocking:

1. **BLOCKING (R4)**: Procrustes residual formula and replacement formula have a transposition bug. Fix as described above — recommend option (a): run Procrustes on original (768, 512) matrices, giving R ∈ R^{512×512}.
2. **Minor (R3)**: Add a 3-seed fallback clause for residuals near the decision thresholds.
