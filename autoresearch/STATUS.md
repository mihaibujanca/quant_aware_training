# Research Status — 91 Cycles (001–091)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 25 |
| Inconclusive | 42 |
| Crash | 1 |

## Latest Cycle
- **Cycle 091** (item 86): MI channel selection — GPT-2 FFN — INCONCLUSIVE (audit corrected from refuted)
  - MI-greedy selects different channels than topk (35-61% overlap)
  - But PPL difference < 1% — practically equivalent
  - Topk heuristic is sufficient; MI adds no value

## Final Answer (consolidated through cycle 091)

### The Recipe
1. **Topk channel selection + lstsq** on probe data (2048+ tokens) — MI-optimal is equivalent
2. **Stack k≤4 blocks at 6×** — ceiling due to multiplicative error compounding (~41%/block)
3. **500 Adam steps** (lr=1e-4, W_down only) → ~+5% PPL
4. No analytic replacement for polish found

### Definitive Findings
- **Basis irrelevant**: SVD = random (071, 077)
- **Channel selection heuristic sufficient**: MI ≈ topk despite 35-61% different channels (091)
- **Piecewise-linear dead**: diverse patterns (085), full-rank Jacobians (088)
- **Error compounds multiplicatively**: ~41%/block, k≤4 ceiling (089)
- **GELU fully nonlinear**: no pruning headroom (083)

### Dead Directions
- Topology, Jacobian SVD, rank-matching, analytic stacking correction
- GELU pruning/clustering, centroid affine, mixed-corpus stacking
- MI channel selection (equivalent to topk, 091)

## Remaining NEXT Items
- **P0**: Item 103 (low-rank Jacobian — low priority given rank95 findings)
- **P2**: Items 87-88 (Procrustes sharing, learned bottleneck AE)
- Only 3 items — agenda refresh required
