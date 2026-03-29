# Research Status — 92 Cycles (001–092)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 25 |
| Inconclusive | 43 |
| Crash | 1 |

## Latest Cycle
- **Cycle 092** (item 107): Adam polish dissection — GPT-2 FFN — INCONCLUSIVE (audit: UNCERTAIN)
  - rank95(delta) ~390/768, but audit flagged sigma vs sigma² issue (rank may be much lower on energy)
  - residual_frac ~0.60 but confounded by truncation (random baseline ~0.51)
  - PPL: step-0 ~72.6 → step-500 ~58.1 (teacher 57.65)
  - Metrics unreliable; re-run with energy-based rank needed for definitive answer

## Final Answer (consolidated through cycle 092)

### The Recipe
1. **Topk channel selection + lstsq** on probe data — MI-optimal is equivalent (091)
2. **Stack k≤4 blocks at 6×** — multiplicative error compounding ~41%/block (089)
3. **500 Adam steps** (lr=1e-4, W_down only) → ~+5% PPL
4. Polish mechanism unclear — dissection inconclusive (092)

### Definitive Findings
- **Basis irrelevant**: SVD = random (071, 077)
- **Channel selection heuristic sufficient**: MI ≈ topk (091)
- **Piecewise-linear dead**: patterns diverse (085), Jacobians full-rank (088)
- **Error compounds multiplicatively**: ~41%/block, k≤4 ceiling (089)
- **GELU fully nonlinear**: no pruning headroom (083)

### Dead Directions
- Topology, Jacobian SVD, rank-matching, analytic stacking correction
- GELU pruning/clustering, centroid affine, mixed-corpus stacking
- MI channel selection, piecewise-linear compression

## Remaining NEXT Items
- **P0**: Items 103 (low-rank Jacobian), 108 (lower compression ratio stacking)
- **P1**: Items 109 (attention compression v2), 110 (non-uniform compression)
- **P2**: Items 87-88 (Procrustes, learned bottleneck AE)
