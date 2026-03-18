# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 010 | Single-affine collapse — toy ReLU | empirical | confirmed | 3/4 layers r<0.15; deep layers are near-linear on data distribution |
| 009 | Jacobian variance atlas | empirical | refuted (proxy) | 0/12 GPT-2 blocks near-linear; no correlation with compression difficulty |
| 008 | Probe-corpus robustness | empirical | confirmed (proxy) | 12/12 blocks: signals probe-robust |
| 007 | Multi-block accumulation curve | empirical | confirmed (proxy) | Easiest-first outperforms hardest-first by 27-36% |
| 006 | Stable channel census | empirical | refuted (proxy) | Stable channels < 1% in 10/12 blocks |

## Insights
- Benchmark: `gpt2` proxy path locked; nanochat exact-open assets absent.
- Single-block: depth-sensitive. Layer 0 hard, middle/late blocks survive 8-16x.
- Multi-block: ordering matters 27-36%. Sweet spot k=4-5 easiest blocks.
- Stable channels: <1% in most GPT-2 blocks — not the answer for GELU FFNs.
- Probe robustness: all geometry signals stable across disjoint probe halves.
- **Jacobian linearity (GPT-2)**: NO block is near-linear. Multi-region approaches required.
- **Single-affine collapse (toy)**: Deep layers (2-4) of trained ReLU MLPs ARE near-linear (r<0.06). Only L1 is genuinely nonlinear (r=0.43). Contrast with GPT-2 result.

## Next
Eligible P0 items with no unmet deps: 22 (Jacobian k-means toy), 24 (RTG spectral toy), 32 (GELU PWL toy), 34 (region volume Pareto toy), 37 (entropy cascade toy), 39 (grokking toy).
Item 21 (GPT-2 single-affine) now unlocked by cycle 010.

## Dead Ends
- Tropical geometry as direct compression
- Exact stable-channel pruning as primary strategy (<1% on gpt2)
- Single-affine collapse for GPT-2 FFN blocks (cycle 009: all blocks highly nonlinear)

## Open Questions
- How many functionally distinct affine behaviors exist per block? (items 22-23)
- Does RTG spectral structure give a principled merging criterion? (items 24-25)
- Can GELU be approximated as piecewise-linear cheaply? (items 32-33)
- Do large activation regions carry most task accuracy? (items 34-35)
- Does the toy single-affine collapse transfer to GPT-2 at all? (item 21)
