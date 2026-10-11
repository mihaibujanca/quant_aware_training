# Cycle 252 audit — item 262

**Claim**: Block 5 is a structural outlier vs cohort {1,2,4,7,8} on geometric metrics.
**Verdict (claimed)**: REFUTED. **Audit**: PASS.

## Phase 1 — decision rule

Plan: CONFIRMED if |z|>=1.5 on >=2 metrics; REFUTED if |z|<1.0 on all 6; INCONCLUSIVE if exactly 1 outlier. Jacobian residual skipped per plan (5 metrics computed, not 6).

Observed z: std_mean=-0.012, kurt_mean=+0.502, eff_rank=+0.633, cov128=-0.775, cov512=-0.801, mean_cos=+0.625. max |z|=0.801 on cov512. Outlier counts: |z|>=1.5 -> 0, |z|>=1.0 -> 0. Rule fires REFUTED cleanly.

## Phase 2 — quality

- Analysis-only, deterministic (seed=42, n_calib=200, seq_len=128, 25600 tokens). Reproducible.
- Block 5 values are monotone within the 4->8 band: eff_rank 358 (between 310 and 379), cov512 0.381 (between 0.406 and 0.394), std_mean 0.153 (between 0.142 and 0.161). No inflection at block 5.
- Cohort {1,2,4,7,8} is wildly heterogeneous: b2 has eff_rank=45 and cov128=0.98 (topk channel concentration), b1 has cov128=0.73. cohort_std for eff_rank is 132 and for cov128 is 0.378 — these inflate the denominator and crush z-scores. The author correctly flags this in the `interpretation` field. A better cohort would be matched neighbours {4,6,7,8}, but even then block 5 sits on the monotone curve, so the conclusion would not flip.
- Interpretation coherent with cycle 250: block 5 is not geometrically special; the cycle 245 drop-block-5 win was a 6->5 count transition.

## Phase 3 — verdict

PASS. Claim REFUTED is correct under the pre-registered rule and robust to cohort concerns. Interesting because it closes the "block 5 is structurally special" story together with cycle 250, fully reframing cycle 245.

**Open**: cross-block interaction at block 5 output feeding downstream attention.ln_1; cohort selection for single-block outlier tests should use matched neighbours, not mixed early/late.
