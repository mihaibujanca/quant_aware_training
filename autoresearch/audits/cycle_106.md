# Audit: Cycle 106 — Gradient Polish k=5 Stacking — GPT-2 FFN

**Item:** 101
**Claimed outcome:** CONFIRMED
**Audited outcome:** CONFIRMED (with flags)
**Status:** PASS (with flags)

## Phase 1: Verdict Check

Original item 101 criteria: "Confirmed if polished k=5 <= +10% PPL for at least 1 config."
Plan-revised criteria: "Confirmed if best polish reduces OWT delta by >=50% for >=1 config."

**Criteria drift.** The plan changed the confirmation threshold from an absolute PPL target (<=+10%) to a relative reduction metric (>=50% reduction). The design review approved this in round 1 under I1 (corpus calibration), reasoning that the absolute +10% threshold was calibrated against Shakespeare-corpus baselines, not OWT. With OWT baseline PPL at 57.65 (vs ~25 on Shakespeare), the 0-step starting point was +58-63%, not +17% as the original item assumed from earlier cycles.

**Assessment of criteria change:** Partially justified. The OWT recalibration is valid — the original +10% target assumed a ~+17% starting point (Shakespeare data), which is not comparable to the +58-63% OWT starting point. However, the relative reduction metric is weaker: 61% relative reduction of a +63% delta still leaves +24% absolute degradation, which is far from the original intent of "polished k=5 is viable." The plan should have either (a) kept absolute thresholds and recalibrated them for OWT, or (b) explicitly acknowledged that this tests "polish efficacy" rather than "k=5 viability."

**Verdict correctness under revised criteria:** Both configs achieve >50% relative reduction at 500 steps (61.3% and 61.2%). CONFIRMED is mechanically correct.

## Phase 2: Quality Checks

### G1: Implementation correctness

**Polish implementation: CORRECT.** Joint W_down optimization on all 5 compressed blocks simultaneously, using Adam at lr=1e-4. Only c_proj.weight (and bias if present) are unfrozen. Model is frozen otherwise (requires_grad_(False)), then selected params re-enabled. Loss is standard cross-entropy on polish_tokens (150 seqs, disjoint from lstsq probe and eval).

**Data splits: CLEAN.** Three disjoint splits:
- Probe train (200 seqs): used for lstsq fitting only.
- Polish/eval holdout (200 seqs): split into 150 polish + 50 eval.
- Eval val (200 seqs): used for final PPL evaluation.

Polish data is disjoint from eval data. No leakage.

**Per-step overfitting monitor: CORRECT.** Every 10 steps, eval PPL on the 50-seq held-out split. Flag if eval exceeds best * 1.02. No flags triggered across all runs.

### G2: Seeds

3 seeds [42, 123, 7]. **PASS for count but FLAG for variance.**

Seed 42 is a massive outlier at 500 steps:
- Config [7,5,1,3,6]: seed 42 = +47.6%, seeds 123/7 = +9.6%/+15.7%
- Config [7,5,4,1,6]: seed 42 = +44.4%, seeds 123/7 = +8.5%/+14.4%

Seed 42 produces 3-5x higher delta than the other seeds. The mean (+24.3%) is heavily skewed by this outlier. The median (+15.7%) and mean-excluding-42 (+12.6%) tell a very different story — they would actually meet the original <=+10% threshold (for seed 123 individually).

This variance is not from polish randomness (the Polish RNG is seeded to 42 in all runs). It comes from the compression procedure: topk selection and lstsq fitting with different probe corpus shuffles. The fact that seed 42 starts at +85% unpolished (vs +46% for seed 123) means the compression quality varies enormously with data sampling.

### G3: Results interpretation

At 500 steps, **mean deltas are +24.3% and +22.4%** — still substantial degradation. For comparison:
- k=4 polished (cycle 061, 500 steps): **+4.9% PPL** (83.9% relative reduction)
- k=5 polished (cycle 106, 500 steps): **+22-24% PPL** (61% relative reduction)

Going from k=4 to k=5 costs ~18pp of absolute PPL degradation even after polish. The k=4→k=5 step incurs ~5x the damage that k=3→k=4 does post-polish. This strongly suggests the k=5 cascade wall is not fully solvable by W_down-only polish.

The 200-step mark already achieves ~52% relative reduction, meaning the last 300 steps contribute only ~9pp additional relative improvement. Diminishing returns are setting in.

### G4: Criteria drift assessment

**FLAG: Criteria drift is concerning but not disqualifying.** The revised criteria correctly address the corpus mismatch problem. However:

1. The 61% relative reduction sounds impressive, but +24% absolute PPL is poor for practical use.
2. Cycle 061 achieved 84% relative reduction for k=4 with the same method. The k=5 polish is substantially less effective (61% vs 84%).
3. The original item's intent was "is k=5 viable after polish?" The answer by the original criteria (<=+10%) is: only for the best seed (seed 123: +9.6% and +8.5%). On average, no.
4. The CONFIRMED verdict is accurate for the revised hypothesis ("does polish work at all for k=5") but would be INCONCLUSIVE under the original criteria (mean +22-24%, original inconclusive range was +10-15%).

### G5: What did we learn

1. **Polish works for k=5 but is less effective than for k=4.** 61% relative reduction vs 84% at k=4. The cascade is harder to correct with more blocks.
2. **Massive seed variance.** The compression procedure (topk + lstsq) produces wildly different starting points depending on the probe data shuffle. Seed 42 starts at +85% while seed 123 starts at +46%. This 40pp spread dwarfs the polish effect.
3. **Best-case k=5 is viable.** Seed 123 reaches +8.5-9.6% after 500 steps — within the original +10% threshold. The problem is reproducibility, not fundamental limitation.
4. **The cascade wall is partially attention-driven.** Cycle 104 showed attention mediates 70%+ of cascade amplification. W_down-only polish cannot address attention-mediated error.

### G6: Method failures

No method failures. The experiment ran cleanly, no NaN/inf, no overfitting flags, eval PPLs monotonically improve (even though per-batch train loss is noisy — expected for batch_size=4).

**Minor agenda inaccuracy:** The outcome text says "Train loss monotonically decreasing (minor non-monotonicity in 1 run first 10 steps only)." In reality, ALL runs show significant non-monotonicity throughout (50% of step transitions are non-decreasing). This is expected for stochastic mini-batch training and is not alarming, but the claim is incorrect. The eval PPL curves are monotonically decreasing, which is the metric that matters.

## Summary

The experiment is well-designed, correctly implemented, and cleanly executed. The CONFIRMED verdict is mechanically correct under the revised criteria. However, the criteria drift from "k=5 viable at <=+10%" to "polish achieves >=50% relative reduction" changes the question being answered. The absolute PPL outcome (+22-24% mean) is far from viable, though the best seed reaches +8.5%. The massive seed variance (40pp spread in starting quality) is the most actionable finding — compression quality depends more on probe data than on the polish procedure.

**Flags:**
- Criteria drift from absolute to relative thresholds (approved by design review but changes the question)
- Seed 42 outlier (3-5x higher delta than other seeds)
- Agenda outcome overstates train loss monotonicity
