# Cycle 097 Brief — Block-0-Skip Attention Stacking (GPT-2)

**Item:** 114 | **Type:** empirical | **Depends on:** none

## Context
Cycle 095 found block 0 catastrophically sensitive (+102% PPL) under rank-11 QKV SVD,
while blocks 1,3,4,9,10 pass individually at ≤+5%. However, simultaneous compression
of top-3 blocks gave +13% — the same as rank-16 — suggesting an inter-block cascade
bottleneck (same c≈0.41 pattern seen in FFN compression, cycles 089/093/094).

## Question
Does explicitly skipping block 0 allow 5-block simultaneous attention compression to
stay within +10% PPL? Is the 3-block +13% ceiling driven by block 0 leakage or by
the same structural cascade rate that limits FFN stacking?

## Approach
Compress 5 viable blocks (rank-11 QKV SVD, skip block 0) simultaneously. Measure PPL.
Secondary: position-dependent block-0 sensitivity probe (pos 0 vs pos 512).

## Risk
If the cascade rate is structural (not block-0-specific), 5-block stacking will exceed
3-block regardless of block 0 — yielding INCONCLUSIVE or REFUTED.

## Assets
- `experiments/attention_compression_v2_gpt2.py` (SVD compression + eval infra)
- `data/openwebtext_sample.txt`
