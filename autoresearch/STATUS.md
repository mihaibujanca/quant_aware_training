# Research Status — 82 Cycles (001–082)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 21 |
| Inconclusive | 37 |
| Crash | 1 |

## Latest Cycle
- **Cycle 082** (item 93): True Gauss-Newton correction retry — GPT-2 FFN — INCONCLUSIVE
  - True GN J^T J via jvp+vjp correctly implemented (CG residuals monotone-decreasing, confirming PSD)
  - GN +21.5% ≈ steepest descent +21.6% ≈ unpolished +21.7% — no improvement
  - Root cause: severely underdetermined (1016 tokens << 98K params), ~97K-dim null space
  - Adam 500 still dominates at +7.7% by iteratively exploring the null space
  - Closes single-step Newton line for GPT-2 FFN compression

## Final Answer (consolidated through cycle 082)

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
- **Single-step Newton fundamentally limited**: n_tokens << n_params makes GN ≈ steepest descent (cycle 082)

### Dead Directions
- Topology (5 experiments), Jacobian SVD basis, rank-matching
- Learned compressors, cascaded re-probing
- Analytic stacking correction via retained-spectrum projection (cycle 080)
- Single-step Newton/Gauss-Newton correction (cycles 081, 082)

## Remaining NEXT Items
- **P1**: Item 94 (GELU near-linearity census v2)
- **P2**: Items 86-88, 95-96 (MI channel selection, Procrustes sharing, learned bottleneck AE, Jacobian clustering, mixed-corpus stacking)
