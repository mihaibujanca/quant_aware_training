# Item
321 (P0): Rank-1 compression for all concentrator blocks — extend cycle 292 to blocks 2, 3, 10, 11.

# Context
- Cycle 292: block 1 rank-1 retrain → 50.08 (1,536 params, Pareto anchor).
- Cycle 289: block 10 rank-5 → 51.47.
- Open: can blocks 2, 3, 10, 11 also work at rank-1?

# Dependencies
None.

# Key Files
- experiments/block1_rank1_cycle292.py — direct template, just swap block index.

# Constraints
- For each block ∈ {2, 3, 10, 11}: rank-1 c_proj weight-SVD, retrain B factor (1, 768) + bias.
- 1,536 trainable params per block.
- 3 seeds each (or 1 seed for budget, 3 seeds if any clears float).
- 4000 steps, best-state restore, disjoint splits.
- Verdict per item 321.

# Handoff
Plan agent: adapt cycle 292 for blocks {2, 3, 10, 11}. New script `experiments/rank1_all_concentrators_cycle311.py`.
