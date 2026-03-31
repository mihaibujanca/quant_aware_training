# Cycle 171 Brief — Item 181: Wider-Shallower Compression

## What
Test whether compressing MORE blocks at LIGHTER per-block ratios yields lower stacked PPL
than the current 4-block-at-6x recipe. Since cycle 163 showed error accumulates diffusely
(no Pareto structure), spreading compression thinly across all 12 blocks may reduce cascade damage.

## Configs
- **Baseline**: 4 blocks [1,4,7,11], 512 hidden (6x), PPL ~70.7
- **A**: 12 blocks, 1024 hidden (3x per block)
- **B**: 12 blocks, 1536 hidden (2x per block)
- **C**: 8 blocks [0..7], 768 hidden (4x per block)
All: topk+lstsq, 500 Adam polish, 3 seeds, 200 OWT eval seqs. Then test corrections (r=4 uniform).

## Decision criteria
- Confirmed: 12@3x PPL < 4@6x PPL (70.7) by >= 2 points
- Refuted: 12@3x PPL >= 4@6x PPL
- Inconclusive: within 2 PPL

## Prior overlap
Cycle 094 tested non-uniform allocation across SAME 4 blocks (REFUTED: allocation irrelevant).
This is different: uniform-but-lighter across MORE blocks. No direct overlap.

## Key risk
12-block probing + polish is 3x the compute of 4-block. Budget ~15 min on MPS.
