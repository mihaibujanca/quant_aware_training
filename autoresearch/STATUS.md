# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 021 | Grokking — toy | empirical | refuted | Pattern count INCREASES during grokking; no compression catalyst |
| 020 | Manifold curvature — GPT-2 | empirical | refuted (proxy) | Curvature flat; LayerNorm confound |
| 019 | Entropy cascade — toy | empirical | confirmed | cH drops 96% L1→L3/L4; correlates with compressibility |
| 018 | Procrustes linearity — GPT-2 | empirical | inconclusive | 7/12 layers lin≥0.98; block 6 identity +3.9% ppl |
| 017 | Region volume Pareto — toy | empirical | inconclusive | Gini=0.408; top 20% covers only 48% |

## Summary of All 13 Cycles (009-021)
| Outcome | Count | Strongest Signals |
|---------|-------|-------------------|
| Confirmed | 2 | Cycle 010 (toy affine), 019 (toy entropy) |
| Refuted | 6 | 009, 011, 012, 014, 020, 021 |
| Inconclusive | 5 | 013, 015, 016, 017, 018 |

## Next
Eligible P0: **40** (attention head census GPT-2, dep 9 done).
After that, need to check items 27, 29, 31, 33, 43, 45 for deps.

## Dead Ends (13 items now)
- Tropical, stable-channel, single-affine (GPT-2), k-means/spectral on maps, intrinsic dim, coarse PWL, volume pruning, curvature (LayerNorm), grokking as simplification (continuous tasks)
