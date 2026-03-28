# Research Status — 79 Cycles (001–079)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 22 |
| Inconclusive | 33 |
| Crash | 1 |

## Latest Cycle
- **Cycle 079** (item 91): Subspace alignment v2 (correct hooks) — GPT-2 FFN — INCONCLUSIVE
  - All 12 blocks: alignment 0.74–0.79 (below 0.8 threshold, 0/12 pass)
  - Random baseline: 0.637. Excess: 0.28–0.41 (non-trivial but moderate)
  - Corrected result invalidates cycle 069 (bug inflated alignment by ~0.05-0.10)
  - FFN spectral structure is partially but not globally shared

## Final Answer (consolidated through cycle 079)

### The Recipe
1. **Random projection + lstsq** on **mixed-corpus probe data** (2048+ tokens from diverse sources)
2. **Stack k=4 blocks at 6×** → ~+60% PPL cross-domain
3. **500 Adam steps** (lr=1e-4, W_down only) → **+4.9% PPL** (in-domain)

### Definitive Findings
- **Basis is irrelevant**: SVD = random at all compression ratios (cycles 071, 077)
- **Corpus diversity matters**: 19.1% PPL improvement from mixed vs single-corpus refit (cycle 078)
- **FFN spectral structure partially shared**: alignment 0.74-0.79 (cycle 079) — topk reasonable but not optimal
- **Stacking error = retained-spectrum corruption** (77-91%, cycle 072)
- **Gradient polish = retained-spectrum denoising**

### Dead Directions
- Topology (5 experiments), Jacobian SVD basis, rank-matching, learned compressors

## Remaining NEXT Items
- **P2**: Items 85-88, 92 (moonshots only)
- All P0/P1 items exhausted again. Consider agenda refresh if moonshots are not the priority.
