# Cycle 150 Plan: Linear factorization + CE single-block (Item 160)

## PLAN INVALID

**Reason**: Superseded by cycles 148 and 149 which together answer this item's question completely:

1. **Cycle 148** showed: Linear AE (no ReLU) matches ReLU AE within 0.7% at single-block. Both beat topk by ~10% (block 4: 54.3 vs 60.8, block 7: 54.9 vs 62.7). CE training is the active ingredient.

2. **Cycle 149** showed: Random-init rank-64 linear factorization trained under STACKED CE loss achieves PPL 69.76, beating topk+polish (70.70) by 1.3% with 1.6x fewer params.

Item 160 asks "does CE-trained linear factorization beat topk at single-block?" The answer from cycle 148 is YES, by ~10%. Running 160 would replicate this finding on 4 blocks instead of 2, which is incremental rather than new knowledge.
