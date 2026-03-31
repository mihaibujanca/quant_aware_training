# Cycle 160: Block-Adaptive Compression Ratio Handoff

## Item
170. **Block-adaptive compression ratio via per-block sensitivity — GPT-2 FFN**

## Context
- **Baseline**: uniform k=4 (6× compression) across all 12 blocks → PPL ~70.7
- **Best known**: topk+lstsq+corrections → PPL ~59 at 1.6M + 49K params
- **Key finding**: composition barrier (Cycle 159) — local quality ≠ stacking quality
- **Question**: can non-uniform channel allocation (via per-block sensitivity) recover 2+ PPL?

## Task
1. Sweep k∈{2,3,4,5,6} per-block (single-block swaps, measure PPL)
2. Allocate global budget of 48 channels across 12 blocks to minimize stacked PPL
3. Compare adaptive allocation vs uniform k=4 at equal parameter count

## Dependencies
- None. Prior work (Cycle 159, 158) frozen.

## Key Files
- `experiments/nonuniform_compression_gpt2.py` — prior non-uniform baseline (Cycle 094)
- `aleph/lm_geometric_compression.py` — topk+lstsq infrastructure
- FFN blocks: [7, 5, 4, 1] (4-block subset, d_model=768)

## Constraints
- Equal total parameter count as uniform k=4 baseline
- Confirm if: PPL ≥ 2 points better than 70.7
- Refute if: PPL within 1 point of 70.7 or worse

## Handoff
Implement per-block sensitivity sweeps (k∈{2,3,4,5,6}) with single-block swaps. Use learned allocator or greedy search to distribute 48-channel budget. Report stacked PPL vs uniform baseline and per-block allocation strategy.
