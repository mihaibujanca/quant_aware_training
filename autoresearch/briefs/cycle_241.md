# Item
251 (P0): W_out + W_fc trainable joint Adam at 6 blocks — does W_fc help at cascade depth?
BLOCKS=[1,2,4,5,7,8] K=512, train W_out + b_out + W_fc + b_fc jointly via Adam, 4000 steps, 3 seeds, eval-every-100 + best-state restore.

# Context
- Cycle 213 (INCONCLUSIVE): W_out+W_fc 3-block → 46.424 (vs cycle 206 W_out only 46.556, delta=0.13).
- Cycle 221 (INCONCLUSIVE): W_out only 6-block → 59.515 ± 0.094.
- Open question: at 6 blocks, does adding W_fc to the trainable surface break the 59.5 ceiling?

# Dependencies
None.

# Key Files
- experiments/adam_full_correction_cycle213.py — W_out+W_fc joint Adam at 3 blocks (template).
- experiments/adam_6block_earlystop_cycle221.py — 6-block scaffolding.

# Constraints
- BLOCKS=[1,2,4,5,7,8], K=512.
- Trainable: W_out + b_out + W_fc + b_fc on the 6 compressed blocks.
- Trainable params per block: 512×768 + 768 + 768×512 + 3072 = 786,432 + 3,840 = 790,272 ≈ 790K. Wait actually: W_fc selected slice is (768, K=512) = 393,216 params + bias 512 = 393,728. W_out (K, 768) = 393,216 + bias 768 = 393,984. Total per block = 787,712. 6 blocks = 4.73M params.
- N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024].
- Confirmed if best_mean ≤ 58.0.
- Refuted if best_mean ∈ [59.0, 60.5].
- Inconclusive otherwise.

# Handoff
Plan agent: combine cycle 213's W_out+W_fc surface with cycle 221's 6-block setup. New script `experiments/adam_wout_wfc_6block_cycle241.py`. Output JSON + summary.
