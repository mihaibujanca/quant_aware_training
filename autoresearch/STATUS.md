# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 009 | Jacobian variance atlas | empirical | refuted (proxy) | 0/12 blocks near-linear (L<0.6 all); no correlation with compression difficulty |
| 008 | Probe-corpus robustness of geometry signals | empirical | confirmed (proxy) | 12/12 blocks: Spearman ≥ 0.95, Jaccard ≥ 0.82 — signals are probe-robust |
| 007 | Progressive multi-block accumulation curve | empirical | confirmed (proxy) | Easiest-first outperforms hardest-first by 27-36% at k=6-9 |
| 006 | Stable and near-linear channel census | empirical | refuted (proxy) | Stable channels < 1% in 10/12 blocks |
| 005 | Single-block swap localization curve | empirical | confirmed (proxy) | 22/24 blocks survive ≥4x compression; layer 0 fails |

## Insights
- Benchmark: `gpt2` proxy path locked; nanochat exact-open assets absent.
- Single-block: depth-sensitive. Layer 0 hard, middle/late blocks survive 8-16x.
- Multi-block: ordering matters 27-36%. Sweet spot k=4-5 easiest blocks. All-block 16x fails.
- Stable channels: <1% in most blocks — not the answer for GELU FFNs.
- Probe robustness: all geometry signals are highly stable across disjoint probe halves.
- **Jacobian linearity: NO block is near-linear. Every block uses genuinely different local maps per token. Multi-region approaches are required.**

## Next
Eligible P0 items: 20 (single-affine toy), 22 (Jacobian k-means toy), 24 (RTG spectral toy), 32 (GELU PWL toy), 34 (region volume Pareto toy), 36 (Procrustes GPT-2), 37 (entropy cascade toy), 38 (manifold curvature GPT-2), 39 (grokking toy), 40 (attention head census GPT-2), 41-45 (moonshots).

## Dead Ends
- Tropical geometry as direct compression
- Exact stable-channel pruning as primary strategy (<1% on gpt2)
- Single-affine collapse for GPT-2 FFN blocks (cycle 009: all blocks highly nonlinear)

## Open Questions
- How many functionally distinct affine behaviors exist per block? (items 22-23)
- Does RTG spectral structure give a principled merging criterion? (items 24-25)
- Can GELU be approximated as piecewise-linear cheaply? (items 32-33)
- Do large activation regions carry most task accuracy? (items 34-35)
