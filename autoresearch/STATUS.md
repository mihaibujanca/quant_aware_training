# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 017 | Region volume Pareto — toy | empirical | inconclusive | Gini=0.408 (moderate skew) but top 20% covers only 48% |
| 016 | GELU PWL — toy | empirical | inconclusive | K=3 drops 35pp, K=10 drops 2.6% |
| 015 | Intrinsic dimension — GPT-2 | empirical | inconclusive | ID rises through depth |
| 014 | RTG spectral — GPT-2 | empirical | refuted (proxy) | Spectral = k-means |
| 013 | RTG spectral — toy | impl+empirical | inconclusive | Signal but RTG too fragmented |

## Key Narrative
Nine cycles of exploration (009-017) have tested:
- Single-affine collapse: works on toy deep layers, fails on GPT-2
- k-means/spectral clustering on affine maps/Jacobians: no exploitable redundancy
- Intrinsic dimension: rises through depth, no compression signal
- GELU→PWL: needs 10+ pieces, not 3
- Region volume Pareto: moderate skew but insufficient for pruning

**Next priority: Item 36 (Procrustes linearity)** — replicates a PUBLISHED positive result ("Your Transformer is Secretly Linear"). If some GPT-2 layers are near-identity, that's a direct compression signal that doesn't depend on region merging.

## Next
Eligible P0: 36 (Procrustes GPT-2), 37 (entropy cascade toy), 38 (manifold curvature GPT-2), 39 (grokking toy), 40 (attention head census GPT-2).

## Dead Ends
- Tropical, stable-channel, single-affine (GPT-2), k-means/spectral on maps, intrinsic dim, coarse PWL, raw volume pruning
