# Item
291 (P0): 2-block [1, 2] K=512 — does adding block 2 (lowest eff_rank) beat cycle 266 [1,4]?

# Context
- Cycle 266: [1, 4] K=512 → 43.852.
- Cycle 271: [1] K=512 → 43.821.
- Cycle 278: block 2 c_proj output eff_rank = 1.05 (lowest of all blocks, essentially rank-1).
- Open: does [1, 2] beat [1, 4]?

# Dependencies
None.

# Key Files
- experiments/two_block_sweep_cycle266.py — direct template, just SUBSETS=[[1,2]].

# Constraints
- BLOCKS=[1, 2], K=512
- Same hyperparameters: N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024], best-state restore. Disjoint splits.

# Handoff
Plan agent: trivial BLOCKS swap on cycle 266. New script `experiments/two_block_1_2_cycle281.py`.
