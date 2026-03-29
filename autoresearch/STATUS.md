# Research Status — 85 Cycles (001–085)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 23 |
| Inconclusive | 38 |
| Crash | 1 |

## Latest Cycle
- **Cycle 085** (item 97): GELU activation pattern diversity — GPT-2 FFN — REFUTED
  - K* > 256 for all 12/12 blocks (reconstruction error 0.69-1.93 at k=256 vs 5% threshold)
  - 2045/2048 unique binary patterns per block — near-maximum diversity
  - Error plateau/reversal at k=128-256 in 5/12 blocks: piecewise-constant GELU approximation is wrong abstraction
  - Closes piecewise-linear compression via GELU pattern clustering. Item 98 (centroid affine) is dead.

## Final Answer (consolidated through cycle 085)

### The Recipe
1. **Random projection + lstsq** on **mixed-corpus probe data** (2048+ tokens, diverse sources)
2. **Stack k=4 blocks at 6×** → ~+30% PPL on training distribution, ~+60% cross-domain
3. **500 Adam steps** (lr=1e-4, W_down only) → **+4.9% PPL** on training distribution
4. No analytic replacement for gradient polish found — Adam steps remain necessary

### Definitive Findings
- **Basis is irrelevant**: SVD = random at all compression ratios (cycles 071, 077)
- **Corpus diversity matters**: 19.1% improvement from mixed refit (cycle 078)
- **FFN spectral structure partially shared**: alignment 0.74-0.79 (cycle 079)
- **Single-step Newton fundamentally limited**: n_tokens << n_params (cycle 082)
- **GELU nonlinearity fully utilized**: 0.04% dead, 0.01% near-linear (cycle 083)
- **GELU patterns maximally diverse**: K* > 256 all blocks, piecewise-linear compression dead (cycle 085)

### Dead Directions
- Topology (5 experiments), Jacobian SVD basis, rank-matching
- Learned compressors, cascaded re-probing
- Analytic stacking correction (cycles 080-082)
- Dead/near-linear GELU pruning (cycle 083)
- Piecewise-linear via GELU pattern clustering (cycle 085)
- Item 98 (centroid affine map compression) — dead, depends on refuted item 97

## Remaining NEXT Items
- **P0**: Item 99 (FFN output manifold geometry)
- **P1**: Item 96 (mixed-corpus stacking)
- **P2**: Items 86-88 (MI channel selection, Procrustes sharing, learned bottleneck AE)
