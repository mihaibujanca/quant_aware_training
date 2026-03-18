# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 013 | RTG spectral coarsening — toy | impl+empirical | inconclusive | Spectral beats random (+9pp) but RTG too fragmented; acc plateaus at 60% |
| 012 | Jacobian k-means — toy | empirical | refuted | K*/N=1.0; k-means on affine maps finds no redundancy |
| 011 | Single-affine collapse — GPT-2 | empirical | refuted (proxy) | 0/12 blocks r<0.15; best swap +33.7% ppl |
| 010 | Single-affine collapse — toy | empirical | confirmed | 3/4 layers r<0.15; deep layers near-linear |
| 009 | Jacobian variance atlas | empirical | refuted (proxy) | 0/12 GPT-2 blocks near-linear |

## Insights
- **Single-affine: works on toy ReLU (deep layers), fails on GPT-2 (every block).**
- **k-means on raw affine maps: no functional redundancy found (cycle 012).** The problem may be the metric, not the redundancy.
- **RTG spectral: real signal but fragmented Hamming-1 graph prevents compression (cycle 013).** k-NN Jacobian graphs (item 25) avoid this.
- Items 23, 35, 41 blocked (depend on item 22). Item 25 still eligible (24 not refuted).

## Next
Eligible P0 items: 25 (spectral GPT-2 — deps met), 26 (intrinsic dim GPT-2), 32 (GELU PWL toy), 34 (region volume Pareto toy), 36 (Procrustes GPT-2), 37 (entropy cascade toy), 38 (manifold curvature GPT-2), 39 (grokking toy), 40 (attention head census GPT-2).

## Dead Ends
- Tropical geometry, stable-channel pruning, single-affine collapse (GPT-2), k-means on raw MASO maps

## Open Questions
- Does k-NN Jacobian graph spectral clustering work on GPT-2? (item 25) ← follows from cycle 013
- Do large activation regions carry most task accuracy? (item 34)
- Is there a Pareto structure in the activation region space? (item 34)
- Can GELU be approximated as PWL? (item 32) — enables ReLU theory on transformers
