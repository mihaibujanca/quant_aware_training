# Cycle 111 Brief — Depth Gradient of GELU Operating Regime (GPT-2 FFN)

**Item:** 122 | **Type:** analysis (interpretability) | **Depends on:** none

## Context
Cycle 103 found the negative fraction of bump-zone activations increases from 54%
(block 0) to 64% (block 11). Cycle 108 showed early blocks have tight pre-activation
concentration (sigma < 0.5) while late blocks approach Gaussian. Cycle 109 confirmed
the negative leak is 87-99% constant bias. STATUS notes "middle blocks nearly linear"
(cycle 107) and "early active GELU concentration" (cycle 108).

## Question
Is there a systematic, monotonic drift toward the negative GELU regime with depth?
If so, deeper blocks are progressively more "gated" (GELU acts as soft binary mask).
Secondary: does the attention+layernorm preceding each FFN contribute to this drift?

## Key concern
Cycle 103 already showed 54% -> 64% over 12 blocks, so the trend likely exists.
The real question is whether it's monotonic and strong enough (slope >0.5%/block,
R^2 > 0.7) or noisy. Must measure on full pre-activation distribution, not just
bump-zone subset. Must also measure E[GELU'(z)] per block as a complementary metric.

## Assets
- `experiments/preactivation_anatomy_gpt2.py` — hooks, pre-activation collection
- Cycle 103/108 results in `autoresearch/results/`
