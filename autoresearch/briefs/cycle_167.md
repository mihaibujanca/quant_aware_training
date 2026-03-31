# Item 177: Sign-Pattern Census on Actual Data — GPT-2 FFN

## Item
Binary sign of z (z > 0 = active): for each token, record which of 3072 channels are positive. On 4096 tokens per block: (a) count unique sign patterns, (b) Hamming distance distribution, (c) cluster patterns with k-means k={16,32,64,128}, measure within-cluster FFN output variance, (d) K* = min k for <5% within-cluster output variance.
- **Confirmed**: K* <= 64 for >=6/12 blocks.
- **Refuted**: K* > 256 for majority of blocks.
- **Inconclusive**: varies across blocks.

## Context
- Cycle 165: negativity is LEARNED — 83-98% channels anti-aligned with E[x], Wx_mean ~ -0.9.
- Cycle 166: negative regime is functionally a constant bias (11/12 blocks within +5% PPL).
- With ~91% channels locked negative, only ~9% (~277) flip per token. This suggests sign patterns are highly redundant — few unique patterns relative to 2^3072.
- Item 183 (sign-pattern-conditioned piecewise lstsq) depends on this census.

## Dependencies
None. ~10 min compute.

## Key Files
- `experiments/constant_neg_bias_gpt2_cycle166.py` — recent pattern for hooking GPT-2 FFN.
- `autoresearch/results/cycle_165_alignment.json` — anti-alignment statistics.
- `autoresearch/results/cycle_166_neg_bias.json` — constant bias result.

## Handoff
Hook mlp.c_fc output for pre-GELU z, binarize (z > 0), collect 4096 patterns per block. Compute unique count, Hamming distances, k-means clustering, within-cluster FFN output variance. K* = min k where within-cluster variance < 5% of total variance.
