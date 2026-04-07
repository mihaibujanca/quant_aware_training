# Item
302 (P0): Block 1 rank-1 c_proj weight-SVD retrain.
Under-eff_rank test like cycle 289 did for block 10.

# Context
- Cycle 280: block 1 rank-2 retrain → 49.03 (2,304 params, Pareto winner).
- Cycle 278: block 1 eff_rank = 1.94 (essentially rank-1).
- Cycle 289: block 10 rank-5 (below eff_rank 16.70) → 51.47 (first sub-eff_rank success).
- Open: block 1 rank-1 (below eff_rank 1.94) might match or beat rank-2.

# Dependencies
None.

# Key Files
- experiments/block1_lowrank_cycle280.py — direct template, retrain at r=1.

# Constraints
- Block 1 c_proj rank-1. Trainable = 1×768 + 768 = 1,536 params.
- 3 seeds retrain, 4000 steps, best-state restore, disjoint splits.

# Handoff
Plan agent: adapt cycle 280 for r=1 retrain. New script `experiments/block1_rank1_cycle292.py`.
