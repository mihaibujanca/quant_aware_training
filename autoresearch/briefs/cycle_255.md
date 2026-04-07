# Item
265 (P0): 4-block K=512 sweep — try [1,2,4,7], [1,4,7,8], [1,2,7,8].
Test whether 4-block compression at K=512 is strictly better than cycle 245's 5-block (53.826).

# Context
- Cycle 210: 3-block [1,4,7] K=512 → 46.556.
- Cycle 245: 5-block [1,2,4,7,8] K=512 → 53.826 (statistical tie with float).
- Open: 4-block sits between 3 and 5; per cycle 245's count interpretation, expected ~50-52 PPL.

# Dependencies
None.

# Key Files
- experiments/drop_block5_cycle245.py — direct template, change BLOCKS.

# Constraints
- CONFIGS = [[1,2,4,7], [1,4,7,8], [1,2,7,8]]
- K=512, N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024], best-state restore.
- Per agenda: phase 1 single-seed sweep, phase 2 3-seed rerun if winner clears 50.0.
- Confirmed if any 4-block config best_mean ≤ 49.0 (cleanly between 3-block 46.556 and 5-block 53.826, monotone in count).
- Refuted if all configs ≥ 53.0.
- Inconclusive otherwise.

# Handoff
Plan agent: design via cycle 245 template + 3 BLOCKS configs. New script `experiments/four_block_sweep_cycle255.py`.
