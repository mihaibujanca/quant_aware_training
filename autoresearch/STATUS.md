# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 012 | Jacobian k-means — toy ReLU | empirical | refuted | K*/N=1.0; k-means on affine maps finds no functional redundancy |
| 011 | Single-affine collapse — GPT-2 | empirical | refuted (proxy) | 0/12 blocks r<0.15; best swap +33.7% ppl |
| 010 | Single-affine collapse — toy ReLU | empirical | confirmed | 3/4 layers r<0.15; deep layers near-linear |
| 009 | Jacobian variance atlas | empirical | refuted (proxy) | 0/12 GPT-2 blocks near-linear |
| 008 | Probe-corpus robustness | empirical | confirmed (proxy) | All geometry signals probe-robust |

## Insights
- Toy ReLU deep layers linearize (cycle 010), but GPT-2 does not (cycles 009, 011).
- **k-means on raw MASO affine maps fails to find functional redundancy (cycle 012).** 375 patterns in a 2×32 spirals network are all "different" under Euclidean distance in map space.
- This does NOT necessarily mean regions are non-redundant — k-means on raw maps may be the wrong metric. Graph-based or output-agreement-based merging (items 24, 34) may succeed where k-means fails.
- Items 23, 35, 41 are now blocked (depend on item 22 confirmed).

## Next
Eligible P0 items: 24 (RTG spectral toy — no deps), 32 (GELU PWL toy — no deps), 34 (region volume Pareto toy — no deps), 36 (Procrustes GPT-2 — dep 9 done), 37 (entropy cascade toy — no deps), 38 (manifold curvature GPT-2 — dep 9 done), 39 (grokking toy — no deps), 40 (attention head census — dep 9 done).

Priority: item 24 (RTG spectral) tests graph-based merging, the natural next step after k-means failure.

## Dead Ends
- Tropical geometry
- Exact stable-channel pruning (<1% on gpt2)
- Single-affine collapse for GPT-2 FFN blocks
- k-means on raw MASO affine maps (no functional redundancy found)

## Open Questions
- Does RTG spectral structure give a better merging criterion than k-means? (item 24)
- Do large activation regions carry most task accuracy? (item 34)
- Can GELU be approximated as piecewise-linear? (item 32)
