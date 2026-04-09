# Item 377: Mixed compression — topk+lstsq for hard blocks, aggressive pruning for easy blocks (R4)

## Context
Cycle 368 inherits refuted dependencies (374, 376) but salvages the block importance ranking [1,7,2,6,9,10,8,3,5,4,11,0] (ascending PPL delta) from cycle 367. Binary pruning is ruled out — all 12 blocks are task-critical (min delta=+1.60 PPL, block 1; catastrophic delta=+3021 PPL, block 0). Frobenius-proportional k allocation is worse than uniform (67.63 vs 66.34 PPL, cycle 365).

## Dependencies
- **374 (refuted)**: Adaptive k by Frobenius is actively harmful; don't replicate.
- **376 (refuted)**: No prunable blocks, but importance ranking is valid.

## Key Files
- `experiments/gpt2_block_importance_item376.py` — importance rank order
- `autoresearch/results/cycle_365_adaptive_k.json` — uniform k=512 baseline (PPL 66.34 ± 0.13)
- `autoresearch/results/cycle_366_residual_error_monitor.json` — error explosion at b10/b11

## Constraints
- Working recipe: topk+lstsq (k=512) + MLP adapter (h=64) on 3 blocks = PPL 47.43; 6 blocks = PPL 65.54 (composition barrier).
- Threshold: mixed PPL < 58 (below progressive 65.54, above single-block 47.43).
- Budget constraint: 3 blocks × 512 = 1536 channels per-block is fixed; reallocation must preserve total budget or justify increase.

## Handoff
Allocate k non-uniformly using importance ranking [1,7,2,6,9,10,8,3,5,4,11,0]: assign lowest k (256–384) to least important blocks [b1, b7, b2, b6], highest k (768–1024) to most important blocks [b11, b0, b4, b5]. Test on 6-block stack [b2,b1,b10,b11,b7,b6] with mixed k vs uniform k=512. Measure 6-block PPL; if PPL < 58, this becomes new working baseline for 12-block scaling (item 360).
