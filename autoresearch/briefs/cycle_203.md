# Cycle 203 Brief: Block Subset Search (Item 214)

## Item

214 — Block subset search — more compressible blocks from {2,3,5,6,8,9,10}

## Context

- Cycle 202: BLOCKS=[1,4,7] K=512 → 61.75 PPL (+14.6% above float). New best recipe.
- Hypothesis: adding some of {2,3,5,6,8,9,10} (all rank-1 input, individually near-redundant)
  might further reduce PPL.
- Cycle 187: individual drops of blocks 1-10 all PPL <40, but 4-block drop is 53 PPL (superadditive).

## Result

REFUTED. No superset of [1,4,7] achieves within 2 PPL of 61.75. Adding 2 blocks costs +13-40 PPL.
3 blocks is the practical ceiling. See `autoresearch/reviews/cycle_203.md`.
