# Cycle 197 Brief: Block-Pair Joint Compression (Item 208)

## Item

208 — Block-pair joint compression — compress consecutive pairs as single units

## Context

The cascade barrier arises from compressing each block independently on float data.
When compressed blocks are stacked, the errors compound. Hypothesis: compressing
block pairs (l, l+1) JOINTLY on the composition f_{l+1}(f_l(x)) might capture
the error interaction and reduce cascade cost.

For each pair (0+1, 2+3, 4+5, 6+7, 8+9, 10+11):
- Train a combined two-block FFN approximation on the joint output
- Compare stacked PPL to single-block sequential compression

## Key question

Does joint compression of consecutive blocks reduce the cascade PPL floor?

## Design

For pair (l, l+1):
- Input: residual stream x before block l
- Target: residual stream output after block l+1 (i.e., f_{l+1}(f_l(x)))
- Compress: fit a SINGLE compressed block (or a pair of compressed blocks) to approximate the joint 2-block transformation

Since a single block with K=512 channels is a limited approximation, we'll test:
1. Each block compressed independently (sequential baseline), then stacked
2. The first block fitted on float data, second block fitted on first-block-compressed output

Method 2 is "sequential progressive fitting" — this directly addresses the cascade by fitting block l+1 on the COMPRESSED block l's output rather than the float output.
