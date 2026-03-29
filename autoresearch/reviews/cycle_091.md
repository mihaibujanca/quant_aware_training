verdict: APPROVE
round: 2
issues: 0 blocking, 1 advisory

## Round 1 fix verification

All 3 fixes applied:

1. **R2 (covariance sample size):** Step 2 now uses 8192 tokens for covariance (N/D=2.67), with 4096 separate for lstsq. P7 updated. Fixed.
2. **R4 (Schur recompute):** Step 4 now recomputes full Schur complement every 50 channels by default. Fallback is recompute-every-step. Fixed.
3. **R1 (Gaussian MI framing):** P6 now explicitly states Gaussian MI-greedy = column subset selection for linear regression, not a general MI estimator. Fixed.

## R1: Gaussian MI soundness

P6 framing is now accurate. Remaining advisory: R1 flagged the cycle 077 dependency (SVD ~ random at k/n=0.67). The plan does not explicitly discuss whether k/n=0.17 escapes that regime. The hypothesis criteria (REFUTED on >90% overlap, INCONCLUSIVE on <3% gap with low overlap) handle verdict classification correctly, so this is not a design flaw — but a one-line note in P2 acknowledging cycle 077's finding at k/n=0.67 would strengthen the motivation for testing at k/n=0.17. Advisory only.

## R2: Covariance sample size

Fixed. 8192 tokens is the bare minimum from R1 (recommended 16384). N/D=2.67 is tight but viable given the regularization guard (1e-6*I if cond>1e10). Acceptable.

## R3: Seeds and statistical power

No change needed, no change made. 3 seeds, 24 measurements, deterministic selection. Fine.

## R4: Greedy tractability

Fixed. Periodic recompute now default. Monotonicity check retained.

## R5: Known pitfalls

MP3/MP6/MP7/MP8 all addressed. No new pitfalls from revisions.

## Summary

All blocking and non-blocking fixes from round 1 applied correctly. Plan is sound. One advisory: acknowledge cycle 077 k/n=0.67 finding in P2 as motivation context for testing at k/n=0.17.
