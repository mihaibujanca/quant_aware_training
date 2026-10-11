# Item
241 (P0): K=128 Adam correction at 6 blocks — does the flat curve hold at higher block count?
BLOCKS=[1,2,4,5,7,8] K=128 Adam W_out+b_out, 4000 steps, 3 seeds, eval-every-100 + best-state restore.

# Context
- Cycle 223 (CONFIRMED): K=128 at 3 blocks → 48.989 vs float 53.895 (-9.10%).
- Cycle 221 (INCONCLUSIVE): K=512 at 6 blocks → 59.515 ± 0.094.
- Cycle 230 just confirmed K=1 still works at 3 blocks (59.245).
- Open question: does the cascade barrier cost the same regardless of K? If yes, K=128 6-block ≈ K=512 6-block ≈ 59.5. If no, K=128 6-block is much worse than K=512 6-block.

# Dependencies
- Item 226 / cycle 221 (done) — joint Adam at 6 blocks K=512.
- Item 228 / cycle 223 (done) — K=128 at 3 blocks.

# Key Files
- experiments/adam_6block_earlystop_cycle221.py — direct template, just swap K.

# Constraints
- BLOCKS=[1,2,4,5,7,8]
- K=128
- Same N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024]
- Best-state snapshot+restore.
- Confirmed if best_mean ≤ 61.0 (within +1.5 of cycle 221's 59.515).
- Refuted if ≥ 65.0 (+5.5 worse — cascade depth amplifies compression cost).
- Inconclusive if 61-65.

# Handoff
Plan agent: trivial K-swap on cycle 221 template. New script `experiments/adam_6block_k128_cycle231.py`.
