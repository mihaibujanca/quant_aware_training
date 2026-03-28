# Research Status — 81 Cycles (001–081)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 21 |
| Inconclusive | 36 |
| Crash | 1 |

## Latest Cycle
- **Cycle 081** (item 85): Analytic Newton correction via Jacobian chain rule — GPT-2 FFN — INCONCLUSIVE (audit: UNCERTAIN — method failure)
  - Implementation used finite-diff full-Hessian (indefinite) instead of true Gauss-Newton J^T J
  - Newton step diverged +902%, but steepest descent +14.3% shows gradient signal is valid
  - Audit downgraded from REFUTED: wrong Hv approximation, 10 CG iters on 98K params, damping ~1e-10
  - Not evidence against the hypothesis — implementation was inadequate

## Final Answer (consolidated through cycle 081)

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
- **P2**: Items 86-88 (moonshots: MI channel selection, Procrustes sharing, learned bottleneck AE)
- Item 85 (Newton correction) remains inconclusive — could be revisited with correct GN implementation
