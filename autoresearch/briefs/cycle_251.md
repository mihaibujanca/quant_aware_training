# Item
261 (P0): 5-block [1,2,4,7,8] at K=128 — does cascade tolerance survive lower channel count?
Apply cycle 245's 5-block configuration but at K=128 instead of K=512.

# Context
- Cycle 245: 5-block [1,2,4,7,8] K=512 → 53.826 (statistical tie with float).
- Cycle 250: 6-block [1,2,3,4,7,8] K=512 → 60.023 (count is the binding constraint).
- Cycle 231: 6-block [1,2,4,5,7,8] K=128 → 67.562 (vs K=512 59.515; +8 PPL).
- Open: 5-block at K=128 — should give us the K=128 penalty applied to cycle 245's near-float baseline.

# Dependencies
None.

# Key Files
- experiments/drop_block5_cycle245.py — direct template, change K to 128.
- experiments/adam_lower_k_cycle223.py — K=128 reference at 3 blocks.

# Constraints
- BLOCKS=[1,2,4,7,8], K=128
- Same hyperparameters: N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024], best-state restore.
- Disjoint calib/eval.
- Confirmed if best_mean ≤ 56.0 (within +2 PPL of float, K=128 penalty stays small at 5 blocks).
- Refuted if best_mean ≥ 62.0 (K=128 penalty scales like 6-block did).
- Inconclusive if 56-62.

# Handoff
Plan agent: trivial K swap on cycle 245 template. New script `experiments/five_block_k128_cycle251.py`.
