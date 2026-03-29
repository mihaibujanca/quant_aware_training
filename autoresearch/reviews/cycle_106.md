verdict: APPROVE
round: 2

## Issue status from round 1

**I1 (corpus calibration): RESOLVED.** Phase 1 now runs 0-step k=5 on OWT as a calibration pass (1 seed) before the polish sweep. All thresholds are re-derived as relative improvement (>=50% confirmed, <20% refuted) over the OWT 0-step baseline. The +30-40% expected range is well-motivated from k=4 OWT-vs-Shakespeare scaling.

**A1 (W_down-only caveat): RESOLVED.** Plan explicitly states in both P6 and the Hypothesis section that W_down-only refutation does NOT close item 101, and that attention-inclusive polish is the motivated follow-up. Fallbacks section reiterates.

**A2 (overfitting logging): RESOLVED.** Step 6 adds per-step logging every 10 Adam steps on a held-out 50-seq split, with an explicit >2% eval PPL regression flag. This is the correct safeguard for the MP3 underdetermined regime (490K params from 25.6K tokens).

## New observations (non-blocking)

- Per-seed reporting (step 8) addresses the round 1 note about seed-42 variance. Good.
- The 50-seq overfitting probe is separate from the 200-seq eval set. Clean design.
- No new pitfall matches from `known_pitfalls.md`. MP3 is mitigated by A2. MP6 is subsumed by the I1 fix (relative thresholds can't be invalidated by an incorrect absolute starting point).

## Summary

All round 1 issues addressed. The plan is ready to run.
