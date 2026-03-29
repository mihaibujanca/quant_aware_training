# Audit — Cycle 108

**Item**: 119 — Pre-activation distribution anatomy — GPT-2 FFN [INTERPRETABILITY]
**Verdict**: INCONCLUSIVE
**Audit status**: PASS

## G1: Code quality

Hook placement is correct: `model.transformer.h[i].mlp.c_fc.register_forward_hook(...)` captures the output of c_fc, which is the pre-GELU linear projection (z = W_in @ LN(x) + b_in). This is the right interception point for pre-activation analysis.

Zone fraction computation (lines 170-183) correctly:
- Computes empirical fraction as `(z.abs() < T).float().mean(dim=0)` then averages over channels.
- Computes Gaussian prediction using the per-channel μ and σ via CDF difference: Φ((T-μ)/σ) - Φ((-T-μ)/σ). This properly accounts for non-zero μ (unlike a symmetric ±T/σ formula).
- Relative excess is `(emp - gauss) / gauss * 100`, which is the correct definition.

Importance score computation (lines 141-156) uses `std(h) * ||w_out|| * (|mean(h)| + |mean(deriv)|)`. The manifest correctly notes the partial tautology with the Spearman correlation: low σ channels → low std(h) → lower importance. The negative ρ is thus partially expected. This is flagged but not a bug.

KS test (lines 186-199) uses scipy.stats.kstest against the fitted Gaussian. Seed is fixed (rng=42), 128 channels sampled. Correct.

## G2: Determinism

No stochastic elements beyond the KS channel sample (seeded at 42). Pre-activations come from a fixed corpus. No training, no random initialization. Deterministic.

## G3: Verdict logic

Criteria from agenda:
- Confirmed: ≤5% excess for ≥10/12 blocks → got 4/12 → NO
- Refuted: >20% excess for ≥6/12 blocks → got 3/12 → NO
- Inconclusive: mixed → YES

4 match + 3 excess + 5 intermediate = INCONCLUSIVE is correct. The verdict follows mechanically from the thresholds.

Note: `max_relative_excess_pct` uses `abs()` of relative excess (line 215), meaning blocks where empirical is BELOW Gaussian (negative excess) can still exceed the 5% threshold. Block 0 shows -7.2% relative excess at T=1.0 (empirical below Gaussian), which counts as 7.2% max excess and excludes it from "match." This is a reasonable design choice — it tests whether Gaussian is a good predictor in both directions — and is consistent with the agenda wording "match within 5%."

## G5: What did we learn?

1. **Early-block concentration is real**: Blocks 1-3 show 21-29% more activations in the GELU nonlinear knee than a per-channel Gaussian predicts. This is not explained by variance alone — there is active structural concentration.

2. **Late blocks are Gaussian**: Blocks 7-10 match Gaussian prediction within 5%. The pre-activation distribution becomes "boring" (well-explained by variance) in later layers.

3. **Bump-dominated channels peak early**: 44.4% of block 1 channels have σ<0.5 and |μ|<0.5 (parked in the GELU knee). This drops to near-zero by block 7. Mean σ increases from 0.52 (block 1) to 0.80 (block 11).

4. **Negative ρ(σ, importance)**: Low-σ channels tend to score higher on importance, but this is partially tautological (importance formula includes std(h), which correlates with σ). The negative ρ is strongest in blocks 4-5 (~-0.6) and vanishes at block 10-11. Still, the direction suggests bump-dominated channels are not "dead" — they are compression-relevant.

5. **Compression implication**: Early blocks where the Gaussian model fails are exactly where GELU nonlinearity matters most. Compression methods that assume approximately-linear activation (large σ → GELU ≈ identity) are safe for late blocks but need care for early blocks.

## Flags

None.
