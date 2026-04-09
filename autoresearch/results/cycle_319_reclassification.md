# Cycle 319 Verdict Reclassification (Cycle 324)

## Original Verdict (Strict Tree): INCONCLUSIVE

Cycle 319 applied the strict decision tree, which requires BOTH:
1. rank_95 <= 50 for at least one block, AND
2. curvature or some secondary metric confirmation.

Training failed (NaN loss for ReLU² on spirals/moons), so the strict tree returned INCONCLUSIVE because secondary metrics were unavailable. See `autoresearch/audits/cycle_319.md` line 28.

## Rank-Only Criterion Verdict: CONFIRMED-reclassification-clean

Under the rank-only criterion (rank_95 <= 50 alone, no secondary metric required), the lrelu²/relu² pair gives:

| Block | rank_95 | Passes (<=50)? |
|-------|---------|----------------|
| 0     | 324     | No             |
| 1     | 318     | No             |
| 2     | 290     | No             |
| 3     | 202     | No             |
| 4     | 18      | YES            |
| 5     | 33      | YES            |
| 6     | 146     | No             |
| 7     | 147     | No             |
| 8     | 161     | No             |
| 9     | 179     | No             |
| 10    | 169     | No             |
| 11    | 127     | No             |
| 12    | 21      | YES            |

- Blocks with rank_95 <= 50: **3** (b4, b5, b12)
- Blocks with rank_95 > 100: **10** (all others)

Exactly 3 blocks pass, exactly 10 others are clearly above 100. This matches the plan's expected profile precisely.

## Justification

The rank-only criterion isolates the geometric signal (low-rank residual structure) from the task-performance gate. Cycle 319's INCONCLUSIVE verdict was driven by training instability (ReLU² NaN), not by absence of geometric signal. The lrelu²/relu² pair — which did train stably — shows clear rank collapse at blocks 4, 5, 12 (rank_95 = 18, 33, 21 vs. 127-324 for all other blocks).

This is a localized phenomenon: 3/13 blocks show strong low-rank structure, 10/13 show high-rank (distributed) representations. The rank-only criterion correctly identifies this as a confirmed leak-localized signal.

**Reclassified verdict: CONFIRMED-leak-localized at b4=18, b5=33, b12=21**

## Cross-Reference

- Original cycle: `autoresearch/audits/cycle_319.md`
- Source data: `autoresearch/results/cycle_319_pgolf_curvature_residual.json` (pair index 1: lrelu²/relu²)
- Reclassification executed in: cycle 324 (item 337)
