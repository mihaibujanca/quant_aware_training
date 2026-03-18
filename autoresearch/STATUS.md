# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 008 | Probe-corpus robustness of geometry signals | empirical | confirmed (proxy) | 12/12 blocks: Spearman ≥ 0.95, Jaccard ≥ 0.82 — signals are probe-robust |
| 007 | Progressive multi-block accumulation curve | empirical | confirmed (proxy) | Easiest-first outperforms hardest-first by 27-36% at k=6-9 |
| 006 | Stable and near-linear channel census | empirical | refuted (proxy) | Stable channels < 1% in 10/12 blocks |
| 005 | Single-block swap localization curve | empirical | confirmed (proxy) | 22/24 blocks survive ≥4x compression; layer 0 fails |
| 004 | Benchmark lock-in and asset audit | analysis | confirmed | Proxy gpt2 path locked; nanochat assets missing |

## Insights
- Benchmark: `gpt2` proxy path locked; nanochat exact-open assets absent.
- Single-block: depth-sensitive. Layer 0 hard, middle/late blocks survive 8-16x.
- Multi-block: ordering matters 27-36%. Sweet spot k=4-5 easiest blocks. All-block 16x fails.
- Stable channels: <1% in most blocks — not the answer for GELU FFNs.
- Probe robustness: all geometry signals are highly stable across disjoint probe halves. Single-probe approach is validated.
- MASO/Jacobian-based analysis is the concrete route forward.

## Next
Item 19: Jacobian variance atlas — compute per-block linearity index L and correlate with compression difficulty from cycle 005. Tests whether near-linear blocks are the easy ones.

## Dead Ends
- Tropical geometry as direct compression
- Exact stable-channel pruning as primary strategy (<1% on gpt2)

## Open Questions
- Which signals best predict block compression difficulty? (Jacobian variance, CKA, local complexity)
- Is there a viable operating point at k=4-5 easiest blocks?
- Can single-affine collapse work for near-linear blocks?
