# Item
289 (P2): Cycle 266 [1,4] K=512 + W_out + W_fc trainable — does unfreezing W_fc help at the 2-block Pareto winner?

# Context
- Cycle 266: [1,4] K=512 W_out-only → 43.852.
- Cycles 213/241/248/262: W_fc retraining noop at K=512 3-block, K=128 6-block, K=512 6-block (bias), K=1 3-block (bias). W_fc surface enlargement is consistently noop.
- Open: at 2 blocks (the Pareto winner), is W_fc still a noop?

# Dependencies
None.

# Key Files
- experiments/adam_full_correction_cycle213.py — FullCompressedMLP trainable W_fc slice.
- experiments/two_block_sweep_cycle266.py — 2-block K=512.

# Constraints
- BLOCKS=[1, 4], K=512
- Trainable: W_out + b_out + W_fc_slice (768×512) + b_fc_slice (512) per block
- Per block: 512×768 + 768 + 768×512 + 512 = 787,712. Total 2 blocks: 1,575,424.
- Adam LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore. 3 seeds. Disjoint splits.
- Confirmed (W_fc helps) if best_mean ≤ 43.5.
- Refuted (W_fc noop) if best_mean ∈ (43.6, 44.1).
- Inconclusive otherwise.

# Handoff
Plan agent: combine cycle 213 FullCompressedMLP with cycle 266 2-block. New script `experiments/two_block_wout_wfc_cycle279.py`.
