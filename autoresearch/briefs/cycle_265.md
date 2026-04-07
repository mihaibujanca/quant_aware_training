# Item
275 (P0): K=128 4-block sweep — extend the K=128 curve, find the Pareto winner.
Test [1,2,4,7], [1,4,7,8], [1,2,7,8] at K=128 joint Adam, 3 seeds.

# Context
- Cycle 255: K=512 4-block [1,2,4,7]=49.609 (-4.29 vs float).
- Cycle 251: K=128 5-block → 59.119.
- Cycle 254 (REFUTED): K=128 5-block 8000 steps → 59.018 (ceiling).
- Cycle 223: K=128 3-block → 48.989.
- Open: where does K=128 4-block sit? Expected ~52-55 PPL.

# Dependencies
None.

# Key Files
- experiments/four_block_sweep_cycle255.py — K=512 4-block sweep template.
- experiments/adam_lower_k_cycle223.py — K=128 reference.

# Constraints
- SUBSETS=[[1,2,4,7], [1,4,7,8], [1,2,7,8]]
- K=128
- Same hyperparameters: N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024], best-state restore. Disjoint splits.
- 9 fits total.
- Verdict per agenda item 275 (need to verify).

# Handoff
Plan agent: read agenda item 275 for thresholds. Trivial K swap on cycle 255 template. New script `experiments/four_block_k128_sweep_cycle265.py`.
