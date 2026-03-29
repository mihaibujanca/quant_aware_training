# Research Status — 83 Cycles (001–083)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 22 |
| Inconclusive | 37 |
| Crash | 1 |

## Latest Cycle
- **Cycle 083** (item 94): GELU near-linearity census v2 — GPT-2 FFN — REFUTED
  - Correct methodology (autograd on tanh-approx GELU derivative) on 4128 mixed-corpus tokens
  - Dead channels (|GELU'|<0.01, >95% tokens): 0.04% — essentially none
  - Near-linear channels (|GELU'-1|<0.05, >80% tokens): 0.01% — essentially none
  - 0/12 blocks meet threshold (need ≥6). GELU nonlinearity is fully utilized.
  - Closes dead/near-linear pruning direction for GPT-2 FFN (R2 exact white-box compression)

## Final Answer (consolidated through cycle 083)

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
- **Single-block metrics misleading**: lstsq domain adaptation inflates quality
- **Single-step Newton fundamentally limited**: n_tokens << n_params (cycle 082)
- **GELU nonlinearity fully utilized**: 0.04% dead, 0.01% near-linear — no pruning headroom (cycle 083)

### Dead Directions
- Topology (5 experiments), Jacobian SVD basis, rank-matching
- Learned compressors, cascaded re-probing
- Analytic stacking correction (cycles 080-082)
- Dead/near-linear GELU channel pruning (cycle 083)

## Remaining NEXT Items
- **P2**: Items 86-88, 95-96 (MI channel selection, Procrustes sharing, learned bottleneck AE, Jacobian clustering, mixed-corpus stacking)
