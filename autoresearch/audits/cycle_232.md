# Cycle 232 — Item 242 Audit

**Claim:** REFUTED — hybrid sequential+joint-polish final_mean = 46.911 ± 0.026.
**Criteria:** Confirmed ≤45.8, Refuted ≥46.5, Inconclusive otherwise.
**Verdict of audit:** PASS (REFUTED stands).

## Threshold check
- final_mean = 46.911 ≥ 46.5 → REFUTED (code branch in `run_seed`→aggregate→`verdict` logic is literal: `if final_mean <= 45.8: CONFIRMED elif final_mean >= 46.5: REFUTED`).
- Gap to refutation threshold: +0.411 PPL. Gap to confirmation threshold: +1.111 PPL.

## Sanity checks
- **Phase 1 reproducibility:** post_sequential_mean = 47.253 ± 0.047, identical to cycle 215's sequential result at the same seeds (42/137/2024) and same protocol. Phase 1 is verbatim c215 logic (per-block Adam W_out+b_out, 2000 steps, LR=1e-4, eval_every=100, best-state restore, collapse rollback to pre-block).
- **Init determinism:** init_ppl_std = 3.3e-06 across 3 seeds — lstsq/topk init is deterministic up to float noise. Matches expectation.
- **Zero collapses:** all 9 phase-1 block runs (3 seeds × 3 blocks) and all 3 phase-2 runs return `collapsed=False`. Restore sanity clean.
- **Per-seed finals:** 46.878 / 46.916 / 46.941 — all three above 46.5, spread = 0.063 PPL. Robust refutation, not a lucky-seed artifact.
- **Polish best_step = 1000 in all 3 seeds:** phase 2 is still descending at end of budget. More steps *might* close the 0.36 PPL gap to c210, but that regime degenerates to "joint 4k with a more expensive init".

## Implementation review (experiments/sequential_polish_hybrid_cycle232.py)
- CompressedMLP / collect_mlp_io / fit_block_lstsq / eval_ppl — verbatim from cycle 215, correct.
- Phase 1 freezes all params, unfreezes only current block's `W_out`+`b_out`, asserts trainable count = 768·K+768. Correct.
- Phase 2 freezes all, unfreezes all 3 blocks' `W_out`+`b_out` jointly, asserts 3·(768·K+768). Correct.
- Phase 2 Adam built fresh (not phase-1 state) at LR=5e-5 — this is the documented protocol ("fresh Adam at LR=5e-5"), not a bug.
- Phase 2 rollback compares `best_ppl_p2 > post_sequential_ppl`, correct collapse semantics.
- Aggregate `final_ppl_mean` uses `final_ppl` (= restored-best for non-collapsed seeds), not pre-restore. Correct.

## Conclusion
Genuine refutation. Sequential warm-start is strictly worse than joint-from-lstsq at matched K=512 3-block W_out+b_out (c210: 46.556 vs c232: 46.911, Δ = +0.355 PPL). The "sequential opens a better basin" hypothesis from cycle 215's audit is dead. The 46.5 floor at 3-block K=512 W_out+b_out is now triangulated across four optimization paths (c206/c210, c213, c215, c232); breaking it requires widening the trainable surface or changing the initialization, not another optimizer schedule. Audit PASS.
