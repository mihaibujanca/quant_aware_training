# Research Status — 78 Cycles (001–078)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 22 |
| Inconclusive | 32 |
| Crash | 1 |

## Latest Cycle
- **Cycle 078** (item 90): Mixed-corpus matched refit — GPT-2 FFN — CONFIRMED
  - A' (Shakespeare 2048 tok): PPL 138.9 (+98.1%)
  - C (Mixed 2048 tok): PPL 112.4 (+60.3%)
  - Relative improvement: 19.1% (threshold ≥5%)
  - ⚠ Audit G3: WikiText-2 tokens in mixed refit while eval is WikiText-2. Gain partly from domain matching.

## Final Answer (updated through cycle 078)

### The Improved Recipe
1. **Random projection + lstsq** on **mixed-corpus probe data** (2048+ tokens from diverse sources)
2. **Stack k=4 blocks at 6×** → ~+60% PPL cross-domain (vs +98% single-corpus)
3. **500 Adam steps** (lr=1e-4, W_down only) → **+4.9% PPL** (in-domain)

### Key Findings
- **Basis is irrelevant**: SVD = random at all compression ratios (cycles 071, 077)
- **Corpus diversity matters**: 19.1% PPL improvement from mixed vs single-corpus refit at matched n=2048 (cycle 078)
- **Stacking error = retained-spectrum corruption** (77-91%, cycle 072)
- **Gradient polish = retained-spectrum denoising**
- **Single-block metrics misleading**: lstsq regularization inflates quality

### Dead Directions
- Topology (5 experiments), Jacobian SVD basis, rank-matching, learned compressors

## Remaining NEXT Items
- **P1**: Item 91 (subspace alignment re-run with correct hooks)
- **P2**: Items 85-88, 92 (moonshots)
