# Item
252 (P1): Early-only vs late-only 3-block compression — disentangle cascade depth from block count.
Run two new BLOCKS configs at K=512 joint Adam: [1,2,3] (early consecutive) and [8,9,10] (late consecutive). Compare to cycle 210 [1,4,7] baseline 46.556.

# Context
- Cycle 210 (3 blocks [1,4,7]) → 46.556.
- Cycle 221 (6 blocks [1,2,4,5,7,8]) → 59.515.
- Cycle 233 [3,4,5,6,7,8] → 67.826 (consecutive middle blocks much worse).
- Open: is depth placement the driver, or block count, or consecutive vs spread?

# Dependencies
None.

# Key Files
- experiments/adam_wout_correction_cycle206.py — 3-block joint Adam template.

# Constraints
- BLOCKS_A = [1, 2, 3] (early consecutive)
- BLOCKS_B = [8, 9, 10] (late consecutive, avoiding block 11)
- K=512
- N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100
- 3 seeds (42, 137, 2024)
- Best-state snapshot+restore
- Confirmed if either config ≥ 2.0 PPL worse than 46.556 (placement matters).
- Refuted if both configs within ±0.8 PPL of 46.556 (count dominates).
- Inconclusive otherwise.

# Handoff
Plan agent: design via cycle 206 template with the two BLOCKS configs. New script `experiments/early_vs_late_3block_cycle242.py`.
