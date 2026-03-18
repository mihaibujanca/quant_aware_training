# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 011 | Single-affine collapse — GPT-2 FFN | empirical | refuted (proxy) | 0/12 blocks r<0.15; best swap +33.7% ppl |
| 010 | Single-affine collapse — toy ReLU | empirical | confirmed | 3/4 layers r<0.15; deep layers near-linear |
| 009 | Jacobian variance atlas | empirical | refuted (proxy) | 0/12 GPT-2 blocks near-linear |
| 008 | Probe-corpus robustness | empirical | confirmed (proxy) | 12/12 blocks: signals probe-robust |
| 007 | Multi-block accumulation curve | empirical | confirmed (proxy) | Easiest-first outperforms hardest-first by 27-36% |

## Insights
- Benchmark: `gpt2` proxy path locked; nanochat exact-open assets absent.
- Single-block: depth-sensitive. Layer 0 hard, middle/late blocks survive 8-16x.
- Multi-block: ordering matters 27-36%. Sweet spot k=4-5 easiest blocks.
- Stable channels: <1% in most GPT-2 blocks.
- Probe robustness: all geometry signals stable across disjoint probe halves.
- **Single-affine collapse: CONFIRMED on toy ReLU (deep layers linearize), REFUTED on GPT-2 (no block collapsible).** GPT-2 FFN blocks are fundamentally multi-region — single-affine approximation is insufficient.
- **Multi-region approaches (items 22-25) are now the critical path.**

## Next
Eligible P0 items: 22 (Jacobian k-means toy), 24 (RTG spectral toy), 32 (GELU PWL toy), 34 (region volume Pareto toy), 36 (Procrustes GPT-2), 37 (entropy cascade toy), 38 (manifold curvature GPT-2), 39 (grokking toy), 40 (attention head census GPT-2).

## Dead Ends
- Tropical geometry as direct compression
- Exact stable-channel pruning (<1% on gpt2)
- Single-affine collapse for GPT-2 FFN blocks (cycles 009 + 011)

## Open Questions
- How many functionally distinct affine behaviors exist per block? (items 22-23) ← CRITICAL
- Does RTG spectral structure give a principled merging criterion? (items 24-25)
- Can GELU be approximated as piecewise-linear cheaply? (items 32-33)
- Do large activation regions carry most task accuracy? (items 34-35)
