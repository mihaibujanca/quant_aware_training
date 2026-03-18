# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 018 | Procrustes linearity — GPT-2 | empirical | inconclusive | 7/12 layers lin≥0.98; block 6 identity +3.9% ppl; joint fails |
| 017 | Region volume Pareto — toy | empirical | inconclusive | Gini=0.408; top 20% covers only 48% |
| 016 | GELU PWL — toy | empirical | inconclusive | K=10 drops 2.6%, K=3 drops 35pp |
| 015 | Intrinsic dimension — GPT-2 | empirical | inconclusive | ID rises through depth |
| 014 | RTG spectral — GPT-2 | empirical | refuted (proxy) | Spectral = k-means |

## Key Narrative
**BREAKTHROUGH SIGNAL: Procrustes linearity (cycle 018)** is the strongest compression predictor found. 7/12 GPT-2 layers have near-identity transitions (linearity ≥ 0.98). Block 6 can be identity-replaced at only +3.9% ppl cost. Spearman with compression damage = 0.56. But joint removal compounds non-linearly (+33.6% for top-2 together).

This aligns with cycle 007 (accumulation ordering matters) and cycle 005 (single-block compression is viable for most blocks). The path forward is combining Procrustes ranking with careful sequential removal.

## Next
Eligible P0: 37 (entropy cascade toy), 38 (manifold curvature GPT-2), 39 (grokking toy), 40 (attention head census GPT-2).
Items 27-31 and 33 have various dependencies.

**Recommended**: Item 37 (entropy cascade) for theoretical grounding, or item 38 (manifold curvature) to complement Procrustes.

## Dead Ends
- Tropical, stable-channel, single-affine (GPT-2), k-means/spectral on maps, intrinsic dim, coarse PWL, raw volume pruning
