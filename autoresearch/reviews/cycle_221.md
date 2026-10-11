# Cycle 221 Design Review — Item 226 (P0)

**Decision:** APPROVE

## Summary
Plan reruns cycle 211's 6-block joint Adam W_out/b_out correction (BLOCKS=[1,2,4,5,7,8], K=512, 4000 steps, LR=1e-4) with eval-every-100 + per-seed best-state snapshot/restore across seeds {42, 137, 2024}. Aims to settle the cycle 211 INCONCLUSIVE.

## Checks
- **Preflight (P1 duplicate):** Not a duplicate. Cycle 211 used a single seed, eval every 500, no best-state restore. Cycle 221 adds seed averaging and formal per-eval early stopping — clear methodological delta.
- **Preflight (P2 deps):** None.
- **Preflight (P3 budget):** Tight but in envelope. Cycle 211 consumed most of 30 min for 1 seed × 4000 steps; 3 seeds ≈ 30 min. Fallback to N_STEPS=3000 is sound because cycle 211's best step was 1500 — 3000 fully brackets it. Eval cadence 100 vs 500 adds ~40 evals/seed at 100 seqs × 128 tokens, cheap.
- **Preflight (P4 program/prompts):** No changes.
- **Hard rule (3 seeds):** Satisfied (42, 137, 2024).
- **Verdict thresholds:** ≤58 CONFIRMED, >62 REFUTED, else INCONCLUSIVE. Asymmetric around cycle 211's 59.680 (which falls in the inconclusive band). Appropriate: proper early stopping cannot make best worse than the trajectory's trough, so only inconclusive→confirmed transitions are plausible.

## Pitfall sweep
- MP1–MP5: N/A (no detection threshold / activation regime / fitting / domain / capacity comparison concerns for this rerun).
- **MP6 (verdict misclassification):** Thresholds correctly gate a three-way verdict. Best-PPL is a meaningful metric (varies with training), won't saturate at trivial value. No risk of false REFUTED on uninformative metric.
- MP7–MP8: N/A (no LayerNorm hook / rank measurement).

## Specific concerns
1. **Wall-time risk.** 3 seeds × 4000 steps × 6-block Adam is tight against 30 min. Plan already has the right fallback (3000 steps); if cycle 211 wall-time was measured closer to 10 min/seed the fallback will trigger. Acceptable.
2. **Best-state snapshot memory.** 6 × (768×512 + 768) fp32 ≈ 9.5 MB per snapshot, cloned to CPU. Only the running-best is kept per seed. Bounded and trivial.
3. **Restore sanity check.** Plan asserts `abs(final_ppl_restored - best_ppl) < 1e-3` after copying cached tensors back. Good guard against state-copy bugs.
4. **Seeds only control the optimizer trajectory.** `init_ppl` is deterministic from topk+lstsq (the same calib set), so init should agree across seeds to <2 PPL — plan documents this as a validation check. Correct.
5. **Comparison baselines.** Plan compares to cycle 211 (59.680) and 3-block joint (46.556). Both are relevant anchors.

## Verdict
All preflight checks pass. Known pitfalls swept — none apply. Methodology soundly addresses the cycle 211 gap (no early stopping). Approve for execution.
