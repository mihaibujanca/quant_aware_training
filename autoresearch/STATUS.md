# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 014 | RTG spectral — GPT-2 FFN | empirical | refuted (proxy) | Spectral = k-means; max improvement 1.3% (need 15%) |
| 013 | RTG spectral — toy ReLU | impl+empirical | inconclusive | Signal exists (+9pp vs random) but RTG too fragmented |
| 012 | Jacobian k-means — toy | empirical | refuted | K*/N=1.0; no redundancy in affine map space |
| 011 | Single-affine collapse — GPT-2 | empirical | refuted (proxy) | No block collapsible; best swap +33.7% ppl |
| 010 | Single-affine collapse — toy | empirical | confirmed | 3/4 layers near-linear (r<0.06) |

## Key Narrative
Three approaches to region-merging have now failed:
1. **k-means on MASO affine maps** (cycle 012): no functional redundancy found
2. **RTG Hamming-1 spectral** (cycle 013): signal real but graph too sparse
3. **k-NN Jacobian spectral** (cycle 014): no advantage over k-means on GPT-2

The problem is NOT the clustering algorithm — both Euclidean and graph methods fail. The problem may be that raw Jacobian/affine map similarity is the wrong measure of functional redundancy for compression.

## Next
Eligible P0: 26 (intrinsic dim GPT-2), 32 (GELU PWL toy), 34 (region volume Pareto toy), 36 (Procrustes GPT-2), 37 (entropy cascade toy), 38 (manifold curvature GPT-2), 39 (grokking toy), 40 (attention head census GPT-2).

## Dead Ends
- Tropical geometry, stable-channel pruning, single-affine (GPT-2), k-means on MASO maps, RTG spectral (both toy and GPT-2)

## Open Questions
- Is there a Pareto structure in region volumes? (item 34)
- Are some GPT-2 layers near-identity? (item 36 — Procrustes)
- Can GELU be approximated as PWL? (item 32) — enables all ReLU theory
- Does intrinsic dimension predict compressibility? (item 26)
