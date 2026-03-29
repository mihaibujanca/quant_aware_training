# Research Status — 94 Cycles (001–094)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 27 |
| Inconclusive | 43 |
| Crash | 1 |

## Latest Cycle
- **Cycle 094** (item 110): Non-uniform compression allocation — REFUTED
  - Non-uniform gives +0.24% vs uniform (below 3% threshold)
  - Aggressive non-uniform is slightly WORSE than uniform
  - Confirms: stacking problem is structural, not allocation-dependent

## Key Structural Finding (cycles 089, 093, 094)
The ~41%/block stacking coefficient is:
1. **Ratio-independent** (3×=4×=6×, cycle 093)
2. **Allocation-independent** (uniform=non-uniform, cycle 094)
3. **Order-independent** (forward=reverse, cycle 089)

This is a fundamental property of inter-block error propagation in the residual stream, not a compression artifact.

## The Recipe
1. **Topk + lstsq** — basis, channel selection, and allocation all near-optimal
2. **Stack k≤4 blocks** — structural ceiling, c≈0.41 regardless of configuration
3. **500 Adam steps** → ~+5% PPL
4. Stacking limit is the binding constraint; per-block improvements don't transfer

## Remaining NEXT Items
- **P0**: Item 103 (low-rank Jacobian)
- **P1**: Item 109 (attention compression v2)
- **P2**: Items 87-88 (Procrustes, learned bottleneck AE)
- Only 4 items — refresh needed after next cycle
