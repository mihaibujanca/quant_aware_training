# Cycle 367 Handoff Brief

## Item
**376: Block importance pruning — zero-out FFN contribution per block, measure PPL delta**

## Context
- Previous cycle (366) identified inflection block at position 9: error explodes at blocks 10–11 (growth rate g_10=86.2x). Cumulative error norm rises slowly through b1–b9 (10.0→50.3), then catastrophically at b10–b11 (50.3→159.6).
- Cycle 339 on pgolf showed b12 is free to prune (delta=-0.000004 BPB). Same method now applies to GPT-2.
- Goal: Identify prunable blocks (delta < 0.5 PPL) and shrink the effective composition problem from 12 blocks to fewer hard blocks.

## Dependencies
- None. Independent experiment.

## Key Files
- Working recipe baseline: `experiments/gpt2_residual_error_monitor_cycle375.py` (cycle 366)
- STATUS.md: 3-block recipe PPL 47.43, 6-block catastrophic at 65.54
- Relevant outputs: `autoresearch/results/cycle_366_error_profile.png`, `autoresearch/results/cycle_366_residual_error_monitor.json`

## Constraints
- Test on GPT-2 (12 blocks, not pgolf).
- Measure per-block importance: zero out FFN residual contribution (set post-FFN output to zero before residual add), measure final test PPL.
- Report delta in PPL for each block (b0–b11).
- Threshold: blocks with |delta| < 0.5 PPL are candidates for skipping/aggressive pruning.

## Handoff
For each block, zero its FFN contribution (output_ffn := 0) and measure PPL. Rank blocks by importance. Identify skippable blocks (delta < 0.5 PPL) to reduce composition problem scope. Output: importance ranking + magnitude of PPL delta per block.
