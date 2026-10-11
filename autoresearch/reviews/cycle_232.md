# Cycle 232 / Item 242 — Design Review

**Verdict: APPROVE**

## Summary
Sequential-init (cycle 215 protocol) followed by lower-LR joint polish on BLOCKS=[1,4,7], K=512, 3 seeds. Tests whether the sequential solution provides a basin that joint Adam at LR=5e-5 can exploit without wiping.

## Soundness
- Novel composition: sequential-alone (215: 47.253) and joint-alone (206/210/213: ~46.4-46.6) both tested; the hybrid is untested and was explicitly flagged by cycle 215's audit.
- Phase 1 reproduces cycle 215 verbatim — good sanity check (`post_sequential_ppl_mean ≈ 47.25`).
- Phase 2 uses LR=5e-5 (half of phase 1), appropriate for polishing rather than relearning.
- Collapse rollback and per-seed honest reporting (no silent restore if phase 2 degrades) matches project rules on not overwriting/hiding results.
- 3 seeds with matching seed set (42, 137, 2024) makes comparison with 215/210 direct.

## Thresholds
- CONFIRMED ≤ 45.8, REFUTED ≥ 46.5. The 46.5 refute band correctly anchors to the existing joint-only floor — if hybrid cannot beat joint-alone, the composition adds nothing. Tight but fair.

## Minor notes (non-blocking)
- Trainable count in phase 2 is 3×(768·K+768) ≈ 1.18M — well within GPU budget.
- Wall-time fallback (1500/800) is documented; acceptable.
- Ensure phase 2 `best` tracking compares against phase-1 best, not just within phase 2, so a worse phase 2 is detectable.

## Decision
Approve as written. Proceed to execute.
