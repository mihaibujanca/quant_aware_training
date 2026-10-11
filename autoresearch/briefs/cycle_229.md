# Item
239 (P2): Sequential vs joint Adam at 6 blocks.
Apply cycle 215 sequential per-block Adam recipe to BLOCKS=[1,2,4,5,7,8] K=512 (2000 steps × 6 blocks × 3 seeds, best-state restore per block).

# Context
- Cycle 215 (item 225, INCONCLUSIVE): sequential per-block Adam at 3 blocks → 47.253 ± 0.047 vs joint 46.556 (joint wins by 0.7 PPL). Cycle 215 audit: "Try sequential with a lightweight joint-polish pass at the end... or sequential without best-state restore but with shared LR schedule across blocks, to see if the 0.7 PPL gap to joint can be closed." Also recommended testing at 6 blocks.
- Cycle 221 (item 226, INCONCLUSIVE): joint 6-block Adam → 59.515 ± 0.094 (best_step ~1367), severe overfitting after.
- Hypothesis: sequential's smaller per-step trainable surface may avoid overfitting and beat the joint 59.515 ceiling.

# Dependencies
- 225 (cycle 215, done) — sequential at 3 blocks.
- 226 (cycle 221, done) — joint at 6 blocks.

# Key Files
- experiments/adam_sequential_cycle215.py — sequential template for 3 blocks.
- experiments/adam_6block_earlystop_cycle221.py — 6-block joint template.

# Constraints
- BLOCKS=[1,2,4,5,7,8] (match cycle 221).
- K=512.
- Sequential order: [1, 2, 4, 5, 7, 8] (block index ascending).
- Per block: 2000 Adam steps, EVAL_EVERY=100, best-state snapshot+restore, then move to next block.
- Trainable per block step: only that block's W_out+b_out. Others frozen at their corrected state.
- 3 seeds (42, 137, 2024).
- Same N_CALIB, N_EVAL, LR as cycles 215/221.
- Wall time estimate: 6 × 2000 = 12000 step-equivalents per seed × 3 seeds = 36000 — TIGHT. Each block trains 395K params at K=512.
- Confirmed if best_mean ≤ 58.0.
- Refuted if best_mean ≥ 60.0.
- Inconclusive if 58 < best_mean < 60.

# Handoff
Plan agent: design via reusing cycle 215 + 221. New script `experiments/adam_sequential_6block_cycle229.py`. Output JSON + summary. Compare against cycle 221 joint baseline 59.515 and cycle 215's pattern.
