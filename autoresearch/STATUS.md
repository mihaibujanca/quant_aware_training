# Research Status — 88 Cycles (001–088)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 24 |
| Inconclusive | 40 |
| Crash | 1 |

## Latest Cycle
- **Cycle 088** (item 102): Input-space token clustering + local Jacobian — GPT-2 FFN — REFUTED
  - K=64 reconstruction errors 153-213% (vs 5% threshold) — barely better than k=1 null
  - FFN Jacobians nearly full-rank at every centroid (rank95=319-478 out of 768)
  - Taylor approximation fails: nonlinearity scale matches data spread
  - Combined with cycle 085: piecewise-linear compression is closed for GPT-2 FFN

## Final Answer (consolidated through cycle 088)

### The Recipe
1. **Random projection + lstsq** on **mixed-corpus probe data** (2048+ tokens, diverse sources)
2. **Stack k=4 blocks at 6×** → ~+30% PPL on training distribution, ~+60% cross-domain
3. **500 Adam steps** (lr=1e-4, W_down only) → **+4.9% PPL** on training distribution
4. No analytic replacement for gradient polish found — Adam steps remain necessary

### Definitive Findings
- **Basis is irrelevant**: SVD = random at all compression ratios (cycles 071, 077)
- **Corpus diversity**: +19.1% single-block (078), attenuates to +7.5% stacked (087)
- **Single-step Newton limited**: underdetermined n_tokens << n_params (082)
- **GELU fully nonlinear**: 0.04% dead (083), K* > 256 patterns (085)
- **Piecewise-linear compression dead**: both input-space (088) and output-space (085) clustering fail; Jacobians are near-full-rank (rank95=319-478/768)
- **FFN manifold moderately isotropic**: doesn't explain lstsq robustness (086)

### Dead Directions
- Topology, Jacobian SVD basis, rank-matching, learned compressors
- Analytic stacking correction (080-082)
- GELU pruning (083), GELU clustering (085), centroid affine (098)
- Input-space clustering (088) — closes piecewise-linear compression entirely
- Mixed-corpus stacking (modest effect, 087)

## Remaining NEXT Items
- **P0**: Item 103 (low-rank Jacobian factorization — but rank95 findings suggest low priority)
- **P1**: Items 104 (compositional error accumulation), 105 (attention compression scan)
- **P2**: Items 86-88 (MI channel selection, Procrustes sharing, learned bottleneck AE)
