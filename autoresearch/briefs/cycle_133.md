# Cycle 133 Brief — Direct Regression Hybrid (Item 127)

## Item
127: Direct regression for middle blocks + topk for edge blocks — GPT-2 FFN [HYBRID]

## Context
- Cycle 107 (item 117, INCONCLUSIVE after audit): direct rank-r regression matches topk in middle blocks (4, 7) but loses badly in early/late (1, 11). Per-block, not stacked.
- Cycle 114 (item 125, REFUTED): linearity-adaptive RATIO allocation (more k to edges) gave <1% improvement over uniform after polish. Ratio allocation is dead.
- This cycle tests METHOD selection (regression vs topk per block), not ratio. Different axis from cycle 114, but same underlying linearity gradient.
- Working recipe baseline: topk+lstsq+500 Adam, k=4 blocks at 6x.

## Dependencies
None hard. Cycle 107 results inform block assignments but are not gating.

## Key Files
- `experiments/direct_regression_bypass_gpt2.py` (cycle 107 — adapt for stacking)
- `autoresearch/results/cycle_107_direct_regression.json` (per-block results)
- `autoresearch/results/cycle_114_linearity_adaptive_ratio.json` (ratio allocation REFUTED)

## Constraints
- 3 seeds [42, 123, 7], 200 eval seqs OWT, +500 Adam polish.
- r=256 for direct regression blocks, k=512 for topk blocks (matched params).
- k=4 stacking (blocks [7, 5, 4, 1] or [1, 4, 7, 11] — clarify in plan).

## Handoff
Build stacking pipeline with per-block method selection. Three conditions: all-topk, all-direct, hybrid (direct for 4,7; topk for 1,11). Compare post-polish PPL. Cycle 114's REFUTED result on ratio allocation is a strong prior that polish erases block-level differences — expect similar outcome here.
