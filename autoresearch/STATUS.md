# Research Status — 90 Cycles (001–090)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 25 |
| Inconclusive | 41 |
| Crash | 1 |

## Latest Cycle
- **Cycle 090** (item 105): Attention compression feasibility scan — INCONCLUSIVE (plan rejected)
  - Rank-128 SVD on [768,768] = 3× not 6× compression. Thresholds miscalibrated.
  - Item valid, needs corrected compression calculation

## Final Answer (consolidated through cycle 090)

### The Recipe
1. **Random projection + lstsq** on probe data (2048+ tokens)
2. **Stack k≤4 blocks at 6×** — ceiling due to multiplicative error compounding (~41%/block)
3. **500 Adam steps** (lr=1e-4, W_down only) → ~+5% PPL
4. No analytic replacement for polish found

### Definitive Findings
- **Basis irrelevant**: SVD = random (071, 077)
- **Piecewise-linear compression dead**: diverse patterns (085), full-rank Jacobians (088)
- **Error compounds multiplicatively**: ~41%/block at 6×, k≤4 ceiling (089)
- **GELU fully nonlinear**: no pruning headroom (083)

### Dead Directions
- Topology, Jacobian SVD, rank-matching, analytic stacking correction
- GELU pruning, clustering (085, 088), centroid affine
- Mixed-corpus stacking (modest, 087)

## Remaining NEXT Items
- **P0**: Item 103 (low-rank Jacobian factorization — low priority)
- **P2**: Items 86-88 (MI channel selection, Procrustes, learned bottleneck AE)
- Only 4 items — refresh after next cycle
