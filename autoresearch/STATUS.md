# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 016 | GELU PWL — toy | empirical | inconclusive | K=3 drops 35pp, K=10 drops 2.6%; needs ~10 pieces |
| 015 | Intrinsic dimension — GPT-2 | empirical | inconclusive | ID rises through depth; no compression correlation |
| 014 | RTG spectral — GPT-2 | empirical | refuted (proxy) | Spectral = k-means; no advantage |
| 013 | RTG spectral — toy | impl+empirical | inconclusive | Signal exists but RTG too fragmented |
| 012 | Jacobian k-means — toy | empirical | refuted | No redundancy in affine map space |

## Key Narrative
**Seven consecutive negative/inconclusive results (cycles 009-016)** on the multi-region and geometry-prediction approaches. The positive result was cycle 010 (toy deep layers are near-linear) but it didn't transfer to GPT-2 (cycle 011).

Remaining high-value experiments:
- **Item 36 (Procrustes)**: replicates published result. If GPT-2 layers are near-identity, depth pruning is viable.
- **Item 34 (region volume Pareto)**: tests if few regions carry most accuracy. Orthogonal to clustering approaches that failed.
- **Item 37 (entropy cascade)**: information-theoretic partition redundancy.

## Next
Eligible P0: 34 (region volume Pareto toy), 36 (Procrustes GPT-2), 37 (entropy cascade toy), 38 (manifold curvature GPT-2), 39 (grokking toy), 40 (attention head census GPT-2).

## Dead Ends
- Tropical geometry, stable-channel pruning, single-affine (GPT-2), k-means on MASO maps, spectral clustering, intrinsic dimension, coarse GELU→PWL (<5 pieces)
