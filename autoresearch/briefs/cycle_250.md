# Item
260 (P0): 6-block [1,2,3,4,7,8] — drop 5, add 3.
Compress 6 different blocks (block 5 out, block 3 in). Test whether dropping just block 5 (cycle 245's finding) lets us add another easy block back.

# Context
- Cycle 221: 6-block [1,2,4,5,7,8] → 59.515.
- Cycle 245: 5-block [1,2,4,7,8] → 53.826 (drops the 5.7 PPL gap).
- Open: can we get a true 6-block compression near float by replacing block 5 with block 3?

# Dependencies
None.

# Key Files
- experiments/adam_6block_earlystop_cycle221.py — direct template, swap BLOCKS.

# Constraints
- BLOCKS=[1,2,3,4,7,8], K=512.
- Same hyperparameters: N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024], best-state restore.
- Confirmed if best_mean ≤ 55.0 (close to float, much better than 59.5).
- Refuted if best_mean ≥ 58.0 (similar to cycle 221's 59.5; the gap was a count phenomenon, not a block-5 phenomenon).
- Inconclusive if 55-58.

# Handoff
Plan agent: trivial BLOCKS swap on cycle 221. New script `experiments/six_block_no5_add3_cycle250.py`.
