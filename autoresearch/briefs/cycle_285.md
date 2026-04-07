# Item
295 (P0): Block 2 rank-r c_proj weight-SVD — r ∈ {1, 2, 5, 10}.
Block 2 has the lowest eff_rank (1.05). Test rank-r compression.

# Context
- Cycle 278: block 2 c_proj output eff_rank = 1.05 (lowest, top-1 energy 99.61%).
- Cycle 280: block 1 rank-2 retrain → 49.03 (2,304 params, current lowest-param Pareto winner).
- Cycle 283: block 2 K sweep K=16=49.46 — beats block 1 at matched K.
- Hypothesis: block 2 rank-1 or rank-2 retrain could beat block 1.

# Dependencies
None.

# Key Files
- experiments/block1_lowrank_cycle280.py — direct template.
- experiments/block11_lowrank_cycle270.py — template.

# Constraints
- For r ∈ {1, 2, 5, 10}: no-retrain PPL + for r=2 (or whichever shows promise) retrain.
- Adapt cycle 280 for block 2.
- 3 seeds retrain, 4000 steps, best-state restore. Disjoint splits.

# Handoff
Plan agent: adapt cycle 280 for block 2. New script `experiments/block2_lowrank_cycle285.py`.
