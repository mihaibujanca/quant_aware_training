# Research Status — 86 Cycles (001–086)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 23 |
| Inconclusive | 39 |
| Crash | 1 |

## Latest Cycle
- **Cycle 086** (item 99): FFN output manifold geometry — GPT-2 FFN — INCONCLUSIVE
  - Isotropy ratios 0.004-0.016 (moderate, in inconclusive band [0.01, 0.1])
  - Spectrum structured (steeper than MP), but JL distortion 3-4× tighter than bound
  - Random≈SVD in lstsq not explained by isotropy alone; low discriminative value

## Final Answer (consolidated through cycle 086)

### The Recipe
1. **Random projection + lstsq** on **mixed-corpus probe data** (2048+ tokens, diverse sources)
2. **Stack k=4 blocks at 6×** → ~+30% PPL on training distribution, ~+60% cross-domain
3. **500 Adam steps** (lr=1e-4, W_down only) → **+4.9% PPL** on training distribution
4. No analytic replacement for gradient polish found — Adam steps remain necessary

### Definitive Findings
- **Basis is irrelevant**: SVD = random at all compression ratios (cycles 071, 077)
- **Corpus diversity matters**: 19.1% improvement from mixed refit (cycle 078)
- **Single-step Newton fundamentally limited**: n_tokens << n_params (cycle 082)
- **GELU nonlinearity fully utilized**: 0.04% dead, 0.01% near-linear (cycle 083)
- **GELU patterns maximally diverse**: K* > 256 all blocks (cycle 085)
- **FFN manifold moderately isotropic**: not enough to explain lstsq robustness (cycle 086)

### Dead Directions
- Topology, Jacobian SVD basis, rank-matching
- Analytic stacking correction (cycles 080-082)
- Dead/near-linear GELU pruning (cycle 083)
- Piecewise-linear via GELU clustering (cycle 085), centroid affine (item 98)
- FFN manifold isotropy as explanation for random≈SVD (cycle 086)

## Remaining NEXT Items
- **P1**: Item 96 (mixed-corpus stacking)
- **P2**: Items 86-88 (MI channel selection, Procrustes sharing, learned bottleneck AE)
- Only 4 eligible items — refresh needed after next cycle
