verdict: REJECT
round: 1

## Issue 1 (blocking) — Discarded-subspace projection is trivially full-rank

At 6x compression, 2560 of 3072 hidden neurons are discarded. `W_discard` has shape `(768, 2560)`. Since 2560 >> 768, the column space of `W_discard` generically spans all of R^768. QR gives Q of shape `(768, 768)` with rank 768, so `Q @ Q.T = I`. The projection `E @ Q @ Q.T = E`, meaning `discarded_energy = 100%` and `E_orth = 0` for both corpora, regardless of signal. Measures (b) and (c) are uninformative by construction.

This is the same class of error as MP6 (verdict misclassification on uninformative metrics). The metric cannot distinguish confirmation from refutation because it is saturated.

**Fix**: The interesting geometric question is not whether E lives in col(W_discard) -- it trivially does when discarded neurons outnumber output dimensions. Instead, consider one of:
- (A) Project E onto col(W_kept) where `W_kept = W_orig[:, selected]` has shape `(768, 512)`. The orthogonal complement of col(W_kept) has dimension `768 - rank(W_kept)` which is at most `768 - 512 = 256`. This decomposition is non-trivial: it asks "can the kept neurons' output directions explain the residual?"
- (B) Work in hidden space: project the pseudo-inverse reconstruction `W_orig^+ @ E.T` onto kept vs discarded neuron indices. This avoids the over-spanning issue entirely.
- (C) Keep only measure (a) — total residual SVD rank comparison across corpora — which does not depend on the subspace projection at all and still tests the core hypothesis.

## Issue 2 (minor) — Agenda item describes different method

The agenda item 68 describes projecting onto `col(A_kept)` (activation matrix of kept channels), while the plan projects onto `col(W_discard)` (weight columns of discarded channels). These ask different questions. The plan's weight-column approach is geometrically better-motivated (modulo the blocking issue above), but this divergence should be acknowledged explicitly.

## Remainder

- The cycle 051 round 1 bias subtraction fix is retained. Good.
- The cycle 051 round 2 correction (use weight columns not activation matrix) is correctly applied. Good.
- Measure (a) — total residual SVD — is sound and unaffected by this issue.
- Sample size, block selection, verdict criteria for measure (a) are appropriate.
- Pre-flight checks P1-P7 are satisfied.
