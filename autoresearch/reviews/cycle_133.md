# Cycle 133 Review — Direct Regression Hybrid (Item 127)

Item 127. Reviewer: design-review agent.

## R1: Premise validity — OK with caveat

Cycle 107 established the linearity gradient. However, cycle 114 (ratio allocation) already showed polish erases block-level differences at <1% margin. The premise that METHOD selection survives polish where RATIO allocation did not is plausible but unproven. Acceptable as an empirical test.

## R2: Measurement soundness — OK

All-topk baseline should reproduce the known ~80 PPL from STATUS. Three seeds, 200 eval seqs. Pre- and post-polish measurements both recorded. No issues.

## R3: Capacity and statistical adequacy — MINOR NOTE

r=256 vs k=512 is NOT param-matched (~394K vs ~786K per block). The plan acknowledges this and follows the agenda as written. The asymmetry is not fatal — it actually favors topk (more params), so if hybrid wins it's a clean result. If hybrid loses, the param gap is a confound. Acceptable for a screening experiment.

## R4: Method-architecture alignment — OK

Both methods use hook-based FFN replacement, compatible with stacking. Direct regression (SVD input basis + lstsq output map) is well-tested from cycle 107. No new architectural concerns.

## R5: Known pitfall scan — OK

- Cycle 114 REFUTED is the main risk (polish convergence). Plan correctly flags this.
- No MP1-MP8 violations. Standard pipeline, standard evaluation.

## Summary

Clean plan. One known risk: cycle 114's precedent strongly suggests polish will erase the hybrid advantage. The param mismatch is acknowledged but not blocking. No code-level issues.

**Verdict: APPROVE.**
