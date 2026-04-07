# Cycle 249 Design Review — Item 259

## Decision: APPROVE

## Summary
Trivial K_PER_BLOCK swap on cycle 247 template. Tests whether per-block residual MSE predicts the binding constraint by placing K=512 on the two highest-MSE blocks {7,8} instead of {4,5}.

## Strengths
- Direct, minimal-diff experiment: single dict change on a working template.
- Trainable param count identical to cycle 247 → fair head-to-head comparison.
- Decision thresholds preserved from cycle 247 (61.0 / 65.0), enabling apples-to-apples interpretation.
- Hypothesis is falsifiable: if MSE-driven prediction is wrong, result lands ≥65 or in inconclusive band.
- 3 seeds, disjoint calib/eval, best-state restore — protocol matches the cycle 221/231/247 family.

## Concerns
- Per-block MSE was reported for seed 42 only in the brief; seeds 137/2024 may rank differently. Mitigated by averaging final ppl over 3 seeds.
- Inconclusive band (61–65) is wide; cycle 247 already landed at 64.453, so a "similar" result here would be uninformative for the MSE-driven hypothesis. Acceptable: still narrows the search.

## Required Changes
None.

## Approval
Plan is self-contained, references exact files, preserves protocol, and has clear decision rules. Approved as-is.
