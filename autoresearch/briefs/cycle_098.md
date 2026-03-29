# Cycle 098 Brief — Joint FFN+Attention Compression (GPT-2)

**Item:** 113 | **Type:** empirical | **Depends on:** none

## Context
FFN 6x compression: k=4 blocks [7,5,4,1] with topk+lstsq+500 Adam polish gives +5% PPL (cycle 089).
Attention rank-11 QKV SVD: 3-block [1,4,9] gives +13%, 5-block gives +43% (cycles 095/097).
Cascade rates differ: FFN c~0.41/block, attention c~0.07/block. These were never tested jointly.

## Question
Are FFN and attention compression errors orthogonal (additive PPL deltas) or amplifying
(super-additive)? If additive, joint compression on k=4 blocks could stay below +20%.

## Approach
Three conditions: (A) FFN-only baseline, (B) attention-only baseline, (C) both simultaneously.
Compare PPL_C to PPL_A + PPL_B - baseline to measure interaction.

## Risk
If FFN and attention errors interact through residual stream (both read/write same residual),
joint damage is likely super-additive. The +5% FFN + +13% attention = +18% additive prediction
already sits near the +20% threshold.

## Assets
- `experiments/compositional_accumulation_curve_gpt2.py` (FFN compression infra)
- `experiments/attention_compression_v2_gpt2.py` (attention SVD infra)
