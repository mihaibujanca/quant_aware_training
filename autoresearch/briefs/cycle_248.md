# Item
258 (P2): K=128 + W_out + W_fc full matrix trainable at 6 blocks — does unfreezing W_fc help where bias alone failed?
BLOCKS=[1,2,4,5,7,8] K=128, train W_out + b_out + W_fc + b_fc jointly. Joint Adam LR=5e-5, 4000 steps, 3 seeds.

# Context
- Cycle 231: K=128 6-block W_out-only → 67.562 (REFUTED).
- Cycle 241: K=512 6-block W_out+c_fc bias → 59.533 (~noop on cycle 221's 59.515).
- Cycle 213: 3-block W_out+W_fc → 46.424 (~noop on cycle 206/210's 46.556).
- Open: at K=128 6-block (much harder surface), does unfreezing W_fc help meaningfully?

# Dependencies
None.

# Key Files
- experiments/adam_full_correction_cycle213.py — full correction (W_out+W_fc) template.
- experiments/adam_6block_k128_cycle231.py — K=128 6-block template.

# Constraints
- BLOCKS=[1,2,4,5,7,8], K=128.
- Trainable: W_out + b_out + W_fc + b_fc per block.
- Per-block trainable params: K×768 + 768 + 768×K + K = 128×768 + 768 + 768×128 + 128 = 98,304 + 768 + 98,304 + 128 = 197,504.
  Wait — that doesn't match the agenda's "≈2,458,880 per block × 6 = 14.75M". Let me recompute.
  Agenda says: 768×128 + 128 + 3072×768 + 3072 ≈ 2.46M per block. The 3072×768 term is the FULL c_fc matrix (768→3072). So W_fc is the FULL c_fc matrix (768, 3072), NOT a sliced (768, K) version.
  - W_out: K×768 = 128×768 = 98,304
  - b_out: 768
  - W_fc full: 768 × 3072 = 2,359,296
  - b_fc full: 3072
  - Total per block: 2,461,440
  - 6 blocks: 14,768,640 ≈ 14.77M trainable
- LR=5e-5 (lower than default for large trainable surface).
- N_CALIB=200, SEQ_LEN=128, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024], best-state restore.
- Disjoint calib/eval.
- Confirmed if best_mean ≤ 61.0.
- Refuted if best_mean ≥ 66.0.
- Inconclusive otherwise.

# Handoff
Plan agent: design via combining full-c_fc-trainable surface from cycle 213 with K=128 6-block setup from cycle 231. New script `experiments/k128_wout_wfc_6block_cycle248.py`. Note: K=128 means CompressedMLP keeps the topk-selected (768, 128) c_fc slice, but the agenda's 14M trainable count suggests the FULL c_fc matrix (768, 3072) is being trained — i.e., the topk is for INIT only, but during training we unfreeze the full c_fc. This is unusual; the topk init says "use these K=128 channels" but unfreezing full W_fc during Adam means Adam can train all 3072 channels. Effectively this becomes a full FFN training.
Plan agent should clarify the semantics in the script.
