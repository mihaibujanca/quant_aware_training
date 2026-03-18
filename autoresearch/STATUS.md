# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 015 | Intrinsic dimension — GPT-2 | empirical | inconclusive | ID rises 1.6→7.5 through depth; no compression correlation |
| 014 | RTG spectral — GPT-2 FFN | empirical | refuted (proxy) | Spectral = k-means; no advantage |
| 013 | RTG spectral — toy ReLU | impl+empirical | inconclusive | Signal exists but RTG too fragmented |
| 012 | Jacobian k-means — toy | empirical | refuted | No redundancy in affine map space |
| 011 | Single-affine collapse — GPT-2 | empirical | refuted (proxy) | No block collapsible |

## Key Narrative
**Pattern of negative results**: Cycles 009-015 have tested single-affine collapse, affine map clustering (k-means), graph-based clustering (spectral), and manifold dimension — all fail as compression signals for GPT-2. The residual stream accumulates information (rising ID) rather than compressing it.

**What's left to test**:
- Item 34 (region volume Pareto): whether few large regions carry most task accuracy — different from clustering
- Item 36 (Procrustes): whole-layer linearity (not per-region) — published result to replicate
- Item 32 (GELU PWL): whether GELU ≈ piecewise-linear — enables ReLU theory
- Item 37 (entropy cascade): information-theoretic partition redundancy

## Next
Eligible P0: 32 (GELU PWL toy), 34 (region volume Pareto toy), 36 (Procrustes GPT-2), 37 (entropy cascade toy), 38 (manifold curvature GPT-2), 39 (grokking toy), 40 (attention head census GPT-2).

## Dead Ends
- Tropical geometry, stable-channel pruning, single-affine (GPT-2), k-means on MASO maps, spectral clustering (both toy and GPT-2), intrinsic dimension as predictor
