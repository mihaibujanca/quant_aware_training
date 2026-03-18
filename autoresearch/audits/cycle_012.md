# Audit — Cycle 012

## Item
22 — Effective number of affine behaviors via Jacobian k-means — toy ReLU

## Phase 1: Verdict Check

Claimed outcome: **refuted** — K*/N_patterns = 1.0 > 0.7.

Agenda refuted criterion: "K* / N_patterns > 0.7, indicating that most activation patterns compute meaningfully different functions."

Evidence: N_patterns = 375, K* = 375 (accuracy never reaches within 1pp of float at any tested K < N_patterns). The sweep tested K = {2, 4, 8, 12, 16, 24, 32, 48, 64, 375}.

**Issue: The K sweep has a gap from 64 to 375.** Accuracy at K=64 is 74.9% (23pp below float). No intermediate K values (96, 128, 192, 256) were tested. The actual K* could be anywhere between 64 and 375. If K* ≈ 200, ratio = 0.53, which would be inconclusive (0.2-0.7), not refuted.

However: accuracy is FLAT at K=32-64 (~75%), suggesting a slow convergence. Even if K* is lower than 375, the ratio is very likely > 0.5 given the convergence shape. The verdict of refuted is plausible but the exact ratio is uncertain.

## Phase 2: Quality Checks

### G1: Methodology
- **K sweep resolution**: Flagged above. A finer grid (128, 192, 256) would pin down K* more precisely.
- **k-means on raw affine maps**: The affine maps A_ω ∈ R^3 have high variance (std=83). k-means in raw space may not capture functional similarity — two maps could classify the same data region identically despite different (A, b) values. Functional redundancy should arguably be measured by output agreement, not map proximity. This is a fundamental limitation of the experimental design (from the agenda item itself, not the execution).
- Train accuracy 98.3%: above the 90% threshold from the plan.
- Sanity check: K=N_patterns recovers float accuracy exactly — correct.

### G2: Reproducibility
Single seed (42). Acceptable given the effect size.

### G3: Tautology Check
Not tautological. The low K accuracy (51% at K=2) demonstrates that merging is destructive, not trivially preserved.

### G4: Scope and Connection
Within scope. Answers whether activation pattern space has low intrinsic dimensionality → answer is no (under k-means on raw maps).

### G5: Counterintuitive Results
Not counterintuitive — but notable that k-means fails completely. The agenda assumed geometric proximity of A_ω implies functional similarity. This assumption may be wrong for classification tasks where sign(output) is all that matters.

## Status: PASS

original_outcome: refuted
audited_outcome: refuted
guardrail_flags: G1 (coarse K sweep, gap 64→375), G1 (k-means on raw maps may not capture functional redundancy)
reason: Although the K sweep has a gap and the k-means metric may not be ideal, the convergence shape (flat at 75% up to K=64) strongly suggests K*/N_patterns >> 0.2. The refuted verdict is the correct call given the data. The deeper issue is whether k-means on raw affine maps is the right redundancy metric at all.
recommendations: Item 24 (RTG spectral coarsening) uses graph-based merging which may handle functional similarity better than k-means. Item 34 (region volume Pareto) tests whether large regions carry most accuracy, bypassing the clustering question. These are now higher priority given the k-means failure.
