# Cycle 255 Design Review — Item 265

## Verdict: APPROVE

## Summary

Three 4-block K=512 subsets ([1,2,4,7], [1,4,7,8], [1,2,7,8]) sweep adapted from cycle 245's template with the agenda's two-phase protocol: phase 1 single-seed (42) sweep, phase 2 3-seed rerun only for subsets clearing best_ppl < 50.0. All other knobs (K, calib/eval split, LR, N_STEPS, EVAL_EVERY, topk+lstsq init → joint Adam W_out+b_out, best-state restore) inherited verbatim from cycle 245.

## Scientific Soundness

- Three subsets cover the early/mid/late drop space: drop block 8 ([1,2,4,7]), drop block 2 ([1,4,7,8]), drop block 4 ([1,2,7,8]).
- Anchors are well-chosen: cycle 210 3-block 46.556, cycle 245 5-block 53.826, float 53.895 — bracket the count-driven extrapolation between 49–51.
- Direct test of the count-driven story from cycle 250: if true, 4-block should land ~5 PPL below 5-block; if false, 5-block was the K=512 W_out cascade limit.

## Methodology

- Same disjoint calib/eval split (all_ids[:200]/all_ids[200:300]) as cycle 245 — directly comparable.
- Two-phase protocol respects the 30-min cap: phase 1 ~9 min (3 fits), phase 2 only triggers on promising subsets.
- Phase 2 trigger threshold (50.0) sits below the CONFIRMED threshold (51.0), so any single-seed result that could plausibly land in the CONFIRMED band promotes to 3-seed mean for noise control.
- Phase-1-only subsets contribute their single-seed best_ppl as best_mean for verdict — conservative for REFUTED (a single-seed run already ≥53 is unlikely to pass on average).

## Thresholds

- CONFIRMED if min subset best_mean ≤ 51.0 (item 265 verbatim, count-driven story).
- REFUTED if all 3 subsets best_mean ≥ 53.0 (asymptotic story, 5-block was the limit).
- INCONCLUSIVE in (51.0, 53.0) — count-driven shrinking marginal benefit.
- 2-PPL band is appropriate given the cycle 245 noise scale.

## Risks

- Phase 2 worst-case (all 3 subsets trigger) is ~27 min, at the cap. Acceptable since the trigger is conservative.
- Phase 1-only subsets have no seed std; if a phase-1 best is ~52 (between thresholds, no trigger), the verdict is INCONCLUSIVE — that is the right call given the single-seed evidence.

## Bookkeeping

- New output paths (`cycle_255_*`) — no collision with cycle 245.
- `cycle=255, item=265` stamped in JSON.
- Helpers (CompressedMLP, fit_block_lstsq, eval_ppl, run_seed) reused verbatim from cycle 245.

## Decision

APPROVE. Proceed to execution.
