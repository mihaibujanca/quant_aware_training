# Cycle 136 Review — Pullback-Metric Error Measurement (Item 146)

Item 146. Reviewer: design-review agent.

## R1: Premise validity — OK

Cycle 135 confirmed pullback metric structure exists on GPT-2 (curvature-damage
rho=-0.65, p=0.032). Testing whether G-weighted error predicts PPL damage better
than Frobenius is the direct follow-up. Reusing cycle 135 PPL deltas is correct
and avoids redundant measurement.

## R2: Measurement soundness — OK

Two-pass approach (float forward, compressed forward, subtract) cleanly isolates
FFN error without mid-forward-pass swaps. Both norms are computed on the same
error tensor, so any comparison is apples-to-apples. Spearman is appropriate for
12-point rank correlation (robust to outliers like block 0's large PPL delta).

## R3: Capacity and statistical adequacy — OK

3 seeds x 200 sequences is adequate for error measurement (no learning involved).
The Spearman correlation is over 12 blocks, which limits statistical power — a
0.15 rho gap from 12 points has wide confidence intervals. But this matches the
agenda's criteria, and the experiment is designed as a signal test, not a
publication-grade statistical claim. Acceptable for the research loop.

## R4: Method-architecture alignment — OK

`pullback_norm(v, U)` signature matches the plan's usage. One concern: the plan
should verify that pullback_norm handles 3D tensors (batch, seq, d_model) or
explicitly reshape to 2D before calling. The docstring says it takes a Tensor but
the implementation may assume 2D. Add a shape assertion or reshape step in the
experiment code.

## R5: Known pitfall scan — OK

- Block 0 PPL delta (9.59) is 2x the next highest. This could dominate the
  Spearman ranking. The plan correctly uses rank correlation (not Pearson), which
  mitigates this. No action needed.
- No MP1-MP8 violations. No training, so capacity/underdetermined concerns are N/A.

## Summary

No blocking issues. One advisory note:
1. Verify `pullback_norm` handles 3D inputs or add explicit reshape in the
   experiment script before calling it.

**Verdict: ACCEPT.**
