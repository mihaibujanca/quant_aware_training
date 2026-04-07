# Item
266 (P0): 6-block K=256 — fill the missing K=256 6-block data point.
BLOCKS=[1,2,4,5,7,8] K=256 joint Adam W_out+b_out, 4000 steps, 3 seeds.

# Context
- Cycle 221: 6-block K=512 → 59.515.
- Cycle 231: 6-block K=128 → 67.562.
- Cycle 223: 3-block K=256 → 47.509 (beats float).
- Cycle 248: 6-block K=128 + W_fc trainable → 67.609 (~same as W_out only).
- Open: where does K=256 6-block sit?

# Dependencies
None.

# Key Files
- experiments/adam_6block_earlystop_cycle221.py — direct template.

# Constraints
- BLOCKS=[1,2,4,5,7,8], K=256
- Same hyperparameters: N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024], best-state restore.
- Confirmed if best_mean ≤ 62.0.
- Refuted if best_mean ≥ 65.0.
- Inconclusive if 62-65.

# Handoff
Plan agent: trivial K swap on cycle 221. New script `experiments/six_block_k256_cycle256.py`.
