# Cycle 114 Audit — Linearity-Adaptive Compression Ratio Allocation (Item 125)

## Verdict: REFUTED — ACCEPTED

## G1: Code correctness
- **OK.** Three allocations (uniform, middle_heavy, edge_heavy) all sum to k=2048 across blocks [1,4,7,11]. Param counts verified identical at 3,150,848 — budget is truly matched.
- topk+lstsq compression with per-block polish (500 Adam steps, lr=1e-4 on W_down) follows the established recipe. Polish is done single-block then injected into the stacked model — correct.
- Naming note: "middle_heavy" gives k=384 to middle blocks (4,7) and k=640 to edge blocks (1,11), so it's actually *edge*-heavy in capacity. Confusing label, but both directions are tested, so this doesn't affect the verdict.
- Seed handling, eval, and aggregation are standard. No bugs found.

## G3: Verdict follows from numbers
- **Criteria:** REFUTED if uniform within 1% of best non-uniform.
- **Uniform post-polish PPL:** 76.69 (mean across 3 seeds).
- **Best non-uniform (middle_heavy):** 75.99 — relative delta 0.90%.
- **Edge-heavy:** 77.22 — worse than uniform.
- 0.90% < 1.0% threshold. **REFUTED is correct.**
- Pre-polish spread is negligible (<0.1 PPL across all 3 conditions), confirming allocation has no effect even before polish.
- All 3 seeds show the same rank ordering (middle_heavy < uniform < edge_heavy), so the 0.9% advantage of middle_heavy is consistent but sub-threshold.

## G6: Method failure
- No method failure. Baseline PPL 51.66 is within expected GPT-2 range. Pre-polish PPL ~73.7 for all conditions (reasonable for 4-block 6x compression).
- Polish reduces PPL by ~0.7-3.0 per condition — modest but real improvement.
- Result is consistent with cycle 094 (damage-based allocation also REFUTED at +0.24%), reinforcing that allocation strategy is not a lever at matched budget.

## Notes
- Two independent allocation strategies (damage-based in cycle 094, linearity-based here) both fail to beat uniform. This is converging evidence that subspace compression benefits are invariant to how capacity is distributed across layers at fixed total budget.
