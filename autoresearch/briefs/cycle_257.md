# Item
267 (P1): 5-block K=64 — extend K-vs-PPL curve at 5 blocks below K=128.
BLOCKS=[1,2,4,7,8] K=64, 4000 steps, 3 seeds.

# Context
- Cycle 230: K=64 3-block → 50.781 (-3.11 vs float).
- Cycle 245: K=512 5-block → 53.826 (~float).
- Cycle 254: K=128 5-block → 59.018 (REFUTED, structural ~59 floor).
- Open: at 5 blocks, where does K=64 sit?

# Dependencies
None.

# Key Files
- experiments/five_block_k128_cycle251.py — direct template, just K swap.

# Constraints
- BLOCKS=[1,2,4,7,8], K=64
- N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024], best-state restore. Disjoint splits.
- Confirmed if best_mean ≤ 60.0 (better than K=128 5-block).
- Refuted if best_mean ≥ 65.0.
- Inconclusive if 60-65.

# Handoff
Plan agent: trivial K swap on cycle 251 template. New script `experiments/five_block_k64_cycle257.py`.
