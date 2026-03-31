# Brief — Cycle 163 (Item 173)

## Item
**Per-block error contribution profiling — stacked topk+lstsq — GPT-2 FFN k=4**
- Identify which blocks are dominant cascade error sources in the stack.
- Goal: determine if corrections should be redistributed non-uniformly across 12 blocks.

## Context
- **Single-block PPL**: ~85 (unpolished)
- **Stacked PPL**: ~95 (pre-polish), ~70.7 (post-polish)
- **Gap**: ~25 PPL from composition → massive cross-block error interaction
- Cycle 151 found: corrections are 93–97% cascade compensation (novel subspace dominates; block 7 was 110% novel recovery, 34% error recovery)
- Current recipe: topk+lstsq+corrections = 1.6M params, 49K corrections, PPL ~59

## Dependencies
- Working compressed models from cycles 161–162 experiments (topk+lstsq pipeline)
- Block-0-to-11 stacking measurement framework
- Marginal contribution metric: PPL(0..i) - PPL(0..i-1) for i=1..11

## Key Files
- `experiments/` — topk+lstsq models (cycles 161–162)
- `autoresearch/results/cycle_151_correction_subspace.json` — per-block error/novel fractions (reference for 93–97% cascade)
- Output: `experiments/per_block_stacking_profile_cycle163.py` (new experiment)

## Constraints
- 12 blocks × 10 snapshots (blocks 0..i, i=0..11) = ~120 forward passes per seed
- Likely 3 seeds for stability
- Use unpolished model (post-lstsq, pre-corrections)
- Measure on OWT validation, fixed seed=42 or {42, 123, 456}

## Handoff
Run per-block incremental stacking: compress blocks 0..i, evaluate PPL at each step. Compute marginal contribution per block. Rank blocks by PPL impact. If 3 or fewer blocks contribute >60% of degradation, proposal: redistribute correction budget (uniform → block-weighted). If uniform (<15% each block), conclude cascade is evenly distributed.
