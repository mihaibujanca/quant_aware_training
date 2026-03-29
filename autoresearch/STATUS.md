# Research Status — 87 Cycles (001–087)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 23 |
| Inconclusive | 40 |
| Crash | 1 |

## Latest Cycle
- **Cycle 087** (item 96): Mixed-corpus stacking eval — GPT-2 FFN — INCONCLUSIVE
  - Mixed-corpus refit improves WikiText-2 by +7.5% relative (below +10% threshold)
  - OWT degradation +1.8% (within 3% limit)
  - Single-block benefit (+19%, cycle 078) attenuates in stacking — cross-block error accumulation

## Final Answer (consolidated through cycle 087)

### The Recipe
1. **Random projection + lstsq** on **mixed-corpus probe data** (2048+ tokens, diverse sources)
2. **Stack k=4 blocks at 6×** → ~+30% PPL on training distribution, ~+60% cross-domain
3. **500 Adam steps** (lr=1e-4, W_down only) → **+4.9% PPL** on training distribution
4. No analytic replacement for gradient polish found — Adam steps remain necessary

### Definitive Findings
- **Basis is irrelevant**: SVD = random at all compression ratios (cycles 071, 077)
- **Corpus diversity matters at single-block**: +19.1% (cycle 078), but attenuates to +7.5% in stacking (cycle 087)
- **Single-step Newton fundamentally limited**: n_tokens << n_params (cycle 082)
- **GELU fully nonlinear**: 0.04% dead channels (083), K* > 256 patterns (085)
- **FFN manifold moderately isotropic**: doesn't explain lstsq robustness (cycle 086)

### Dead Directions
- Topology, Jacobian SVD basis, rank-matching, learned compressors
- Analytic stacking correction (cycles 080-082)
- GELU pruning (083), GELU clustering (085), centroid affine (098)
- Mixed-corpus stacking (modest effect, cycle 087)

## Remaining NEXT Items
- **P2**: Items 86-88 (MI channel selection, Procrustes sharing, learned bottleneck AE)
- Only 3 items remain — agenda refresh required
