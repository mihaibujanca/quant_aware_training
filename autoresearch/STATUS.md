# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 004 | Benchmark lock-in and asset audit | analysis | confirmed | Exact-open target is nanochat + climbmix; current machine only has runnable `gpt2` proxy path |
| 003 | Survey: Non-distillation model compression methods (2023-2026) | literature | confirmed | Dead-neuron removal and always-active collapse are the strongest no-SGD exact baselines |
| 002 | Survey: Tropical geometry and neural networks | literature | inconclusive | Tropical polynomial form is not an operational compression path; zonotope clustering is the only salvageable idea |
| 001 | Survey: Balestriero piecewise linear / spline theory | literature | confirmed | Activation census and affine-map extraction are directly actionable from MASO theory |

## Insights
- The benchmark ladder is now locked: `nanochat` + `climbmix-400b-shuffle` is the exact-open target, `gpt2` is the current proxy, `hf_random` is smoke only.
- Exact white-box pruning is the first serious no-SGD baseline; Serra/Berzins-style stable-channel removal should be treated as mandatory, not optional.
- Tropical geometry remains de-prioritized for direct compression; it does not supply a compact constructive representation for deep nets.
- MASO-style local-map reasoning is still the most concrete route to transformer FFN analysis.

## Next
Item 10: Single-block swap localization curve — now that the benchmark policy is explicit, measure where analytic FFN compression fails inside the full model, starting with `gpt2` unless local nanochat assets are added.

## Dead Ends
- Tropical geometry as a direct compression representation
- Treating proxy `gpt2` runs as if they answered the exact-open benchmark

## Open Questions
- How predictive are single-block local metrics of full-model degradation after the swap?
- Do pretrained transformer FFNs contain enough stable or near-linear channels to make exact pruning material?
- When nanochat assets are added, do the proxy conclusions transfer or break?
