# Item
250 (P0): K=1 Adam correction at 6 blocks — does the K=1 floor hold at cascade depth?
BLOCKS=[1,2,4,5,7,8] K=1 Adam W_out+b_out, 4000 steps, 3 seeds, eval-every-100 + best-state restore.

# Context
- Cycle 230 (CONFIRMED): K=1 at 3 blocks → 59.245 ± 0.029. NO compression cliff in [1, 512] at 3 blocks.
- Cycle 221 (INCONCLUSIVE): K=512 6-block → 59.515 ± 0.094.
- Cycle 231 (REFUTED): K=128 6-block → 67.562. K=128 penalty triples from 3→6.
- Open question: does K=1 6-block also collapse like K=128, or hit ~60 like K=512?
- If ~60: structural cascade-depth floor confirmed.
- If catastrophic: channel capacity matters at 6 blocks even though it doesn't at 3.

# Dependencies
None.

# Key Files
- experiments/adam_6block_earlystop_cycle221.py — direct template, just swap K to 1.

# Constraints
- BLOCKS=[1,2,4,5,7,8]
- K=1
- Same N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100
- 3 seeds (42, 137, 2024)
- Best-state snapshot+restore.
- Trainable: 6 × (1×768 + 768) = 9216 total params (vs cycle 221's 2.36M).
- Confirmed if best_mean ∈ [58.0, 62.0] (K=1 ≈ K=512 at 6 blocks → structural floor).
- Refuted if best_mean ≥ 70.0 (K=1 collapses like K=128 did → channel capacity matters at 6 blocks).
- Inconclusive if 62 < best_mean < 70 (partial steepening).

# Handoff
Plan agent: trivial K-swap on cycle 221 template. New script `experiments/adam_6block_k1_cycle240.py`.
