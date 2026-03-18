# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 007 | Progressive multi-block accumulation curve | empirical | confirmed (proxy) | Easiest-first outperforms hardest-first by 27-36% at k=6-9; ordering matters enormously |
| 006 | Stable and near-linear channel census in pretrained FFNs | empirical | refuted (proxy) | On proxy `gpt2`, stable or near-linear FFN channels stayed below `1%` in `10/12` blocks |
| 005 | Single-block swap localization curve | empirical | confirmed (proxy) | On proxy `gpt2`, `22/24` latest `>=4x` single-block swaps met the agenda criterion; only layer `0` failed |
| 004 | Benchmark lock-in and asset audit | analysis | confirmed | Exact-open target is nanochat + climbmix; current machine only has runnable `gpt2` proxy path |
| 003 | Survey: Non-distillation model compression methods (2023-2026) | literature | confirmed | Dead-neuron removal and always-active collapse are the strongest no-SGD exact baselines |

## Insights
- The benchmark ladder is now locked: `nanochat` + `climbmix-400b-shuffle` is the exact-open target, `gpt2` is the current proxy, `hf_random` is smoke only.
- Single-block swaps are sharply depth-sensitive: layer 0 is hard, middle/late layers survive 8x-16x compression.
- Ordering matters enormously for multi-block accumulation: 27-36% perplexity gap between easiest-first and hardest-first at k=6-9. Sweet spot is k=4-5 easiest blocks.
- All-block 16x compression fails regardless of ordering (ppl=403, 3.3x teacher).
- Exact white-box pruning (stable channels) is too weak on proxy gpt2: <1% of channels in most blocks. Not the whole answer.
- MASO-style local-map reasoning and Jacobian-based signals are the most concrete route forward.

## Next
Item 13: Probe-corpus robustness of geometry signals — test whether compression signals (channel rankings, Jacobian spectra, selected hidden-unit sets) are robust across disjoint probe subsets. Note: stable-channel masks are moot given cycle 006 refutation.

## Dead Ends
- Tropical geometry as a direct compression representation
- Treating proxy `gpt2` runs as if they answered the exact-open benchmark
- Exact stable-channel pruning as a primary compression strategy (< 1% on proxy gpt2)

## Open Questions
- Are compression geometry signals robust across different probe subsets?
- Which signals (Jacobian, CKA, local complexity) best predict block compression difficulty?
- Is there a viable operating point at k=4-5 easiest blocks with partial compression?
- Do the proxy conclusions transfer to nanochat once exact-open assets are available?
