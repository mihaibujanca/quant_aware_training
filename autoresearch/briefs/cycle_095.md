# Cycle 095: Attention Compression v2 — GPT-2

## Item
109. Attention block compression feasibility scan v2 — GPT-2.

## Context
Item 105 (cycle 090) was rejected twice: rank-128 SVD on full [768,768] attention matrices = 3x compression, not 6x. Thresholds were calibrated to the wrong ratio.
Item 109 corrects this: operate per-head (12 heads, d_head=64). Each QKV projection is [768,64] = 49,152 params per head. Rank-11 SVD gives (768x11 + 11x64) = 9,152 params = 5.4x compression per head.

## Dependencies
- None. Standalone feasibility scan.
- FFN compression recipe (cycles 089/093/094) provides comparison: c~0.41 stacking coefficient, 6x viable on FFN.

## Key question
Are GPT-2 attention QKV projections low-rank enough for ~6x compression? Per-head rank-11 is the correct target.

## Approach
Single-head swap PPL: for each of 144 heads (12 layers x 12 heads), replace one head's QKV with rank-11 SVD, measure PPL delta. Then test pruning bottom-3 heads + rank-16 SVD on remaining 9.

## Output
`experiments/attention_compression_v2_gpt2.py`, results JSON + plot.
