# Item
278 (P1): K=64 4-block sweep — extend cycle 257 to 4 blocks.
Test [1,2,4,7], [1,4,7,8], [1,2,7,8] at K=64.

# Context
- Cycle 257: K=64 5-block → 62.181.
- Cycle 255: K=512 4-block [1,2,4,7] → 49.609.
- Cycle 265: K=128 4-block [1,2,4,7] → 53.647 (DOMINATED by cycle 223 K=128 3-block).
- Cycle 230: K=64 3-block → 50.781.

# Dependencies
None.

# Key Files
- experiments/four_block_k128_sweep_cycle265.py — direct template, K swap.
- experiments/four_block_sweep_cycle255.py — K=512 4-block sweep.

# Constraints
- SUBSETS=[[1,2,4,7], [1,4,7,8], [1,2,7,8]]
- K=64
- 3 seeds × 3 subsets = 9 fits.
- Same hyperparameters: N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore. Disjoint splits.
- Verdict per agenda item 278 (verify).

# Handoff
Plan agent: trivial K swap on cycle 265 template. New script `experiments/four_block_k64_sweep_cycle268.py`.
