# Item
290 (P0): Block 1 c_proj rank-r weight-SVD sweep — r ∈ {1, 2, 5, 10}.
Block 1 c_proj output eff_rank=1.94 (cycle 278) → rank-2 should capture most of its signal.

# Context
- Cycle 270: block 11 rank-22 weight-SVD + retrain → 54.47 (17.7K params).
- Cycle 278: block 1 c_proj output eff_rank=1.94 (top-1 energy 92.94%).
- Hypothesis: block 1 is easier to compress at the c_proj weight level than block 11.

# Dependencies
None.

# Key Files
- experiments/block11_lowrank_cycle270.py — direct template, swap block index and rank.

# Constraints
- For r ∈ {1, 2, 5, 10}: replace c_proj_1.weight with rank-r SVD. No-retrain PPL first.
- Then for r=2: retrain with Adam (trainable = (r × 768) + 768 = 2304 params for r=2).
- N_CALIB=200, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore. 3 seeds for retrain.
- Disjoint splits.
- Confirmed if any r ≤ 10 no-retrain ≤ 55 OR r=2 retrain ≤ 54.
- Refuted if all r ≥ 60 no-retrain AND r=2 retrain ≥ 55.
- Inconclusive otherwise.

# Handoff
Plan agent: adapt cycle 270 template for block 1 with r ∈ {1, 2, 5, 10}. New script `experiments/block1_lowrank_cycle280.py`.
