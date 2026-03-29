verdict: ACCEPT
round: 2

## Issue status

**I1 (fixed).** Decomposition now subtracts variant D (both-frozen) as passthrough baseline. Fractions `attn_frac + gelu_frac + inter_frac = 1.0` by algebraic construction. Confirmed.

**I2 (fixed).** Variant D (both sub-layers receive float inputs) included as step 3, fourth variant. Cost updated to 132 fwd passes. Confirmed.

**A1 (partially addressed).** Step 6 reports `cos(δ_F, δ_L)` per transition — captures whether the full block rotates the error vector. Would be stronger with per-variant cosines (`cos(δ_A, δ_L)`, `cos(δ_B, δ_L)`) to see which sub-layer is responsible for rotation, but the current level is sufficient for the hypothesis being tested.

**A2 (addressed).** Step 5 reports `||δ_D|| / ||δ_L||` per transition, capturing LayerNorm nonlinear passthrough effect. Logged to results JSON. Sufficient.

## R1: Criteria faithfulness
P2 criteria unchanged from round 1. Verdict logic (step 7) is sound: dominant_count ≥ 8 with same_source → CONFIRMED, all balanced → REFUTED, else INCONCLUSIVE. Pass.

## R2: Confound check (residual concerns)
The decomposition is clean when `amp_full > 0` (net amplification). One edge case: if a specific transition has `||δ_F|| < ||δ_D||` (sub-layers attenuate error), `amp_full` is negative and all fractions flip sign / blow up. Prior work shows c ≈ 0.41 on average, but individual transitions could attenuate. **Advisory A3:** guard the fraction computation — skip transitions where `amp_full < ε` (e.g. ε = 0.01 × ||δ_L||) and report them separately as "no amplification" transitions. This avoids division-by-near-zero artifacts in the stacked bar plot.

## R3: Known pitfalls
- **MP7:** Measurement point is post-FFN, post-residual-add, pre-LN. Correct.
- **MP6:** With I1 fixed, fractions are interpretable. No risk of forced verdict.
- MP1-5, MP8: Not applicable.

## R4: Cost and feasibility
132 fwd passes, 3 seeds, ~12 min MPS. Feasible. No change from round 1 estimate.

## R5: Verdict logic soundness
Sound. The interaction term as residual (amp_full - amp_attn - amp_gelu) properly captures super/sub-additive effects without double-counting.

## Advisory (new)

**A3: Guard against negative amp_full.** If `||δ_F|| ≤ ||δ_D||` at a transition, fractions are undefined. Skip those transitions in the dominance count and report count separately. Does not affect verdict logic if ≤3 transitions are skipped; if more, force INCONCLUSIVE.
