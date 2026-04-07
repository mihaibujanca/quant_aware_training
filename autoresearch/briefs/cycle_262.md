# Item
268 (P2): K=1 + c_fc bias trainable on 3 blocks — does the c_fc bias become useful at the smallest K?
BLOCKS=[1,4,7], K=1, trainable {W_out, b_out, c_fc bias slice (1 scalar per block)} = ~2,313 params. 3 seeds.

# Context
- Cycle 230: K=1 3-block W_out only → 59.245.
- Cycle 213/241/248: c_fc bias is a noop at K=512 and K=128 + W_fc full at K=128.
- Hypothesis: at K=1 the W_out path is starved enough that the bias becomes useful.

# Dependencies
None.

# Key Files
- experiments/adam_k_floor_cycle230.py — K=1 3-block template.
- experiments/adam_full_correction_cycle213.py — FullCompressedMLP class.

# Constraints
- BLOCKS=[1,4,7], K=1
- FullCompressedMLP-style trainable {W_out (1×768=768), b_out (768), c_fc_bias_slice (1 scalar per block)}
- Per block: 768 + 768 + 1 = 1537 (cycle 230 had 1536: 768 + 768; this adds 1 scalar). 3 blocks: 4611.
- Adam LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore. 3 seeds. Disjoint splits.
- Confirmed if best_mean ≤ 58.5.
- Refuted if best_mean ≥ 59.5.
- Inconclusive if 58.5-59.5.

# Handoff
Plan agent: design via cycle 230 + adding c_fc bias slice to trainable. New script `experiments/k1_wfc_bias_cycle262.py`.
