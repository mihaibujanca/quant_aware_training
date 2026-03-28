# Research Status — 80 Cycles (001–080)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 22 |
| Inconclusive | 34 |
| Crash | 1 |

## Latest Cycle
- **Cycle 080** (item 92): Retained-spectrum denoising — GPT-2 FFN — INCONCLUSIVE (audit: UNCERTAIN)
  - Analytic projection is algebraically vacuous: re-solving lstsq with same h_comp ≈ no-op
  - Shakespeare corpus makes results incomparable to cycle 061 (OWT)
  - Adam sub-baseline PPL is domain fine-tuning artifact
  - Retained-spectrum denoising via lstsq is a dead end

## Final Answer (consolidated through cycle 080)

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

### Dead Directions
- Topology (5 experiments), Jacobian SVD basis, rank-matching
- Learned compressors, cascaded re-probing
- Analytic stacking correction via retained-spectrum projection (cycle 080)

## Remaining NEXT Items
- **P2**: Items 85-88 (moonshots: Jacobian chain Newton, MI channel selection, Procrustes sharing, learned bottleneck AE)
