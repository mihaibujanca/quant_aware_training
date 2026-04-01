# Audit — Cycle 190 (Item 200)

**Verdict:** INCONCLUSIVE
**Hybrid all-12 PPL:** 28.980 (+27.67%, baseline 22.700)

## Result Quality

Progressive sweep PPL:
k=0: 23.285 (+2.58%)
k=1: 23.359 (+2.90%)
k=2: 23.652 (+4.19%)
k=3: 24.263 (+6.88%)
k=4: 25.677 (+13.11%)
k=5: 26.967 (+18.80%)
k=6: 28.045 (+23.55%)
k=7: 28.484 (+25.48%)  ← peak
k=8: 28.457 (+25.36%)
k=9: 28.127 (+23.91%)
k=10: 27.851 (+22.69%)  ← local minimum
k=11: 28.980 (+27.67%)

Results are clean. Baseline 22.700 matches cycle 188 exactly — same config, same data.

## Key Findings

1. **Both error types are roughly equal in cost**: Positive-regime shape change alone (+27.67%)
   costs nearly as much as negative-regime constant replacement alone (C3 +20.33%). Neither
   dominates clearly — the 28.98 vs 27.31 difference is within the margin of comparison.

2. **Superadditive interaction confirmed**: C3 (+20%) + hybrid (+28%) costs should add to ~48%
   if independent. But ReLU+bias (both modifications simultaneously) is +88%. This 40% extra
   confirms strong superadditive interaction when BOTH regimes are modified.

3. **Non-monotone progressive**: PPL peaks at k=7 (28.484) then DECREASES to k=10 (27.851)
   before rising again at k=11 (28.980). This is surprising — adding more hybrid blocks
   in positions 8-10 actually IMPROVES PPL relative to the k=7 peak. Possible explanation:
   cascade error direction (cycle 189) — later blocks may partially cancel earlier errors when
   the activation pattern changes symmetrically.

4. **Block 11 anomaly persists**: k=10→k=11 is the largest single-block PPL jump (+1.13 PPL),
   consistent with block 11 anomaly in cycles 184 and 189.

## Comparison Table

| Modification              | All-12 PPL | Pct Increase |
|---------------------------|-----------|--------------|
| Baseline                  | 22.700    | 0%           |
| C3 (neg→constant)         | 27.314    | +20.33%      |
| Hybrid (pos→ReLU)         | 28.980    | +27.67%      |
| ReLU+bias (both modified) | ~42.6     | ~+88%        |

The hybrid result refutes the hypothesis that positive-regime shape change IS the dominant
error — hybrid is only marginally worse than C3, not substantially worse.

## Next Agenda

Both P0 items exhausted. Agenda refresh required to determine new directions. The superadditive
interaction between negative and positive regime modifications is the central unexplained finding.
