# Research Status — 89 Cycles (001–089)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 25 |
| Inconclusive | 40 |
| Crash | 1 |

## Latest Cycle
- **Cycle 089** (item 104): Compositional error accumulation — GPT-2 FFN — REFUTED
  - Error compounds multiplicatively: ~41% PPL growth per additional compressed block (R²=0.933)
  - k=4: +56% PPL, k=5: +145%, k=8: +914% — hard ceiling at ~4 blocks at 6×
  - Block ordering doesn't matter at scale (both orderings converge to identical PPL at k=8)
  - No polish in this run; prior work (cycle 061) showed polish helps at k=4

## Final Answer (consolidated through cycle 089)

### The Recipe
1. **Random projection + lstsq** on probe data (2048+ tokens)
2. **Stack k≤4 blocks at 6×** — practical ceiling due to multiplicative error compounding
3. **500 Adam steps** (lr=1e-4, W_down only) → ~+5% PPL on training distribution
4. Beyond k=4: not viable at 6× compression without error mitigation

### Definitive Findings
- **Basis irrelevant**: SVD = random at all ratios (071, 077)
- **Piecewise-linear compression dead**: patterns maximally diverse (085), Jacobians full-rank (088)
- **Error compounds multiplicatively**: ~41%/block at 6×, k=4 ceiling (089)
- **GELU fully nonlinear**: no pruning headroom (083)
- **Single-step Newton limited**: underdetermined (082)

### Dead Directions
- Topology, Jacobian SVD basis, rank-matching
- Analytic stacking correction (080-082)
- GELU pruning (083), clustering (085, 088), centroid affine (098)
- Mixed-corpus stacking (modest, 087)

## Remaining NEXT Items
- **P0**: Item 103 (low-rank Jacobian — low priority given rank95 findings)
- **P1**: Item 105 (attention compression feasibility scan)
- **P2**: Items 86-88 (MI channel selection, Procrustes, learned bottleneck AE)
- Only 5 items — refresh after next cycle
