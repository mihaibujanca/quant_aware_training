# Research Status — 84 Cycles (001–084)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 22 |
| Inconclusive | 38 |
| Crash | 1 |

## Latest Cycle
- **Cycle 084** (item 95): Jacobian clustering — GPT-2 FFN — INCONCLUSIVE (design review REJECT)
  - Plan had affine linearization inconsistency: d_c = mean(GELU'(z_i)) ≠ GELU'(z_c) due to Jensen's inequality
  - Item is valid — needs re-proposal with consistent linearization (override d_c = GELU'(z_c))
  - Not executed

## Final Answer (consolidated through cycle 084)

### The Recipe
1. **Random projection + lstsq** on **mixed-corpus probe data** (2048+ tokens, diverse sources)
2. **Stack k=4 blocks at 6×** → ~+30% PPL on training distribution, ~+60% cross-domain
3. **500 Adam steps** (lr=1e-4, W_down only) → **+4.9% PPL** on training distribution
4. No analytic replacement for gradient polish found — Adam steps remain necessary

### Definitive Findings
- **Basis is irrelevant**: SVD = random at all compression ratios (cycles 071, 077)
- **Corpus diversity matters**: 19.1% improvement from mixed refit (cycle 078)
- **FFN spectral structure partially shared**: alignment 0.74-0.79 (cycle 079)
- **Stacking error = retained-spectrum corruption** (cycle 072), but analytic correction doesn't work (cycle 080)
- **Single-step Newton fundamentally limited**: n_tokens << n_params (cycle 082)
- **GELU nonlinearity fully utilized**: 0.04% dead, 0.01% near-linear (cycle 083)

### Dead Directions
- Topology (5 experiments), Jacobian SVD basis, rank-matching
- Learned compressors, cascaded re-probing
- Analytic stacking correction (cycles 080-082)
- Dead/near-linear GELU channel pruning (cycle 083)

## Remaining NEXT Items
- **P2**: Items 86-88, 96, 97-99 (MI channel selection, Procrustes sharing, learned bottleneck AE, mixed-corpus stacking, GELU pattern diversity, FFN output manifold geometry)
