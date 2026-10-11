# Cycle 229 Design Review — Item 239 (P2)

## Verdict: APPROVE

## Summary

Direct extension of cycle 215's sequential per-block Adam recipe from 3 blocks to 6 blocks (`BLOCKS=[1,2,4,5,7,8]`, K=512, 3 seeds). Same machinery, same per-block best-state restore, same collapse rollback. Verdict thresholds match item 239 verbatim (CONFIRMED ≤58.0, REFUTED ≥60.0, INCONCLUSIVE in between). Compares against cycle 221 joint 6-block ceiling (best_mean=59.515).

## Strengths

- Recipe is a verbatim port of cycle 215 (already validated machinery): per-block fresh Adam, best-state restore, collapse rollback.
- Three baselines explicitly tracked (joint 6-block 59.515, sequential 3-block 47.253, joint 3-block 46.556) — gives clean attribution if result differs from joint.
- Thresholds (58.0 / 60.0) come straight from item 239 — no drift.
- `n_trainable=393984` assertion catches any freezing bug early.
- `init_ppl ≈ 86.7` cross-check against cycle 221 verifies the compressed model build matches.
- Per-block trajectory reporting (after_b1 / after_b2 / ...) allows diagnosis of where sequential diverges from joint, if at all.
- 3 seeds (42/137/2024) match the cycle 215/221 protocol, allowing apples-to-apples mean±std comparison.

## Risks / Watchpoints

- **Wall time**: stated ~25 min (6 blocks × 2000 steps × 3 seeds). Cycle 221 ran 6-block joint in similar time but joint shares forward passes; sequential redoes the forward pass per block. Plan documents the fallback (reduce to 1500 steps), justified by cycle 215 best step ≈2000 and cycle 221 best step ≈1367 — 1500 covers both. Acceptable.
- **Sequential ordering**: ascending [1,2,4,5,7,8]. Cycle 215 used the same ordering for [1,4,7]. Consistent.
- **Block 1 anchor**: validation expects substantial drop after block 1 (cycle 215: 61.7 → ~50). At 6 blocks the init is ~86.7, so absolute numbers differ but the relative drop should still be visible — minor wording note, not blocking.
- **Collapse handling**: best_ppl ≥ pre_block_ppl ⇒ rollback. Already in the cycle 215 template, just inherited. Fine.

## Recommendations

None blocking. Approve as written.
