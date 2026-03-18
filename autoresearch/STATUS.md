# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 020 | Manifold curvature — GPT-2 | empirical | refuted (proxy) | Curvature flat (0.045-0.047); LayerNorm confound |
| 019 | Entropy cascade — toy | empirical | confirmed | cH drops 96% L1→L3/L4; correlates with compressibility |
| 018 | Procrustes linearity — GPT-2 | empirical | inconclusive | 7/12 layers lin≥0.98; block 6 identity +3.9% ppl |
| 017 | Region volume Pareto — toy | empirical | inconclusive | Gini=0.408; top 20% covers only 48% |
| 016 | GELU PWL — toy | empirical | inconclusive | K=10 drops 2.6%, K=3 drops 35pp |

## Summary of All 12 Cycles (009-020)
| Outcome | Count | Cycles |
|---------|-------|--------|
| Confirmed | 2 | 010 (toy single-affine), 019 (toy entropy cascade) |
| Refuted | 5 | 009 (Jacobian atlas), 011 (GPT-2 single-affine), 012 (k-means), 014 (spectral GPT-2), 020 (curvature) |
| Inconclusive | 5 | 013 (RTG spectral), 015 (intrinsic dim), 016 (GELU PWL), 017 (Pareto), 018 (Procrustes) |

**Strongest signals**: Procrustes linearity (cycle 018) and entropy cascade (cycle 019).

## Next
Eligible P0: 39 (grokking toy, no deps), 40 (attention head census GPT-2, dep 9 done).
Items 27, 29 unlocked by cycle 010 (dep 20). Items 33 unlocked (dep 32 not refuted).

## Dead Ends
- Tropical, stable-channel, single-affine (GPT-2), k-means/spectral on maps, intrinsic dim, coarse PWL, volume pruning, manifold curvature (LayerNorm confound)
