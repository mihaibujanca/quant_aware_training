# Cycle 178 — Plan

**Item:** 188 — Negative-regime bias absorption + positive-only compression

## Pre-flights

1. Load GPT-2, eval baseline PPL on 200 OWT sequences (expect ~22.7). Abort if outside [15, 65].
2. Compute b_leak = E[gelu(z) | z<0] per block per channel (3072-dim) on calibration data (~4096 tokens).
3. Compute positive-regime mask: per block per channel, fraction of calibration tokens where z >= 0. Channels with frac_positive > 0 are "sometimes positive."

## Design

### BiasAbsorbed model
For each of 12 FFN blocks:
- Hook into `mlp.act`: replace GELU(z) with ReLU(z) + b_leak[block]
- b_leak is a 3072-dim constant vector, not learned
- Eval full-model PPL with all 12 blocks absorbed (should match cycle 166 ~42 PPL)

### Positive-only compression (experiment condition)
For blocks [1, 4, 7, 11]:
- Identify positive-regime channels: those with frac_positive > 0.01 (at least 1% of tokens activate)
- These channels (~270-400 per block) are the only ones needing compression
- Apply topk selection within positive-regime channels (k = min(512, n_positive))
- Collect hidden activations (ReLU only, b_leak folded into output bias) and fit lstsq
- The output projection must reconstruct: W_down @ (ReLU(z) + b_leak) = W_down @ ReLU(z) + W_down @ b_leak
- Since b_leak is constant, absorb W_down @ b_leak into the block's residual bias
- Compressed MLP only needs to model the ReLU part on selected channels

### Standard topk+lstsq (control)
Same blocks [1, 4, 7, 11], standard GELU topk k=4 (512 channels), lstsq + 500-step polish. Use existing library.

## Measurements

For each of 3 seeds {42, 137, 7}:
1. `ppl_base`: float baseline
2. `ppl_absorbed_only`: all 12 blocks GELU->ReLU+b_leak, no compression
3. `ppl_positive_compress`: 4 blocks positive-only compressed + 8 blocks absorbed
4. `ppl_standard_topk`: 4 blocks standard topk+lstsq (GELU), 8 blocks float
5. Per-block: n_positive_channels, topk_within_positive, lstsq residual

## Verdict logic

- Mean ppl_positive_compress <= 68 AND < ppl_standard_topk - 2 --> CONFIRMED
- ppl_absorbed_only > ppl_base * 1.05 AND ppl_positive_compress > ppl_standard_topk --> REFUTED
- Otherwise --> INCONCLUSIVE

## Script

`experiments/neg_bias_absorption_compression_cycle178.py`

## Output

- `autoresearch/results/cycle_178_neg_absorption.json`
- `autoresearch/results/cycle_178_neg_absorption.png`
