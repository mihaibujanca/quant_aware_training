# Item
285 (P0): Hybrid — [1] K=512 + block 11 rank-22 c_proj compose?
Combine cycle 271's [1] K=512 with cycle 270's block 11 rank-22 c_proj weight-SVD retrain. Test compose-ability.

# Context
- Cycle 271: [1] K=512 → 43.82 (394K trainable).
- Cycle 270: block 11 rank-22 weight-SVD + retrain → 54.47 (17.7K trainable).
- These operate on DISJOINT blocks. Do they compose?

# Dependencies
- Cycle 270 SVD + retrain machinery.
- Cycle 271 1-block K=512 script.

# Key Files
- experiments/block11_lowrank_cycle270.py — block 11 rank-22 retrain.
- experiments/one_block_sweep_cycle271.py — [1] K=512 sweep.

# Constraints
- Block 1 compressed via K=512 topk+lstsq.
- Block 11 c_proj replaced with rank-22 SVD (U_22 Σ_22 frozen, B=Vh[:22, :] trainable, 22×768 = 16,896 + 768 bias = 17,664 trainable).
- Joint Adam trainable: block 1 W_out (512×768=393,216) + b_out (768) + block 11 B (16,896) + c_proj_11 bias (768) = 411,648 total.
- LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore. 3 seeds. Disjoint splits.
- Confirmed if best_mean ≤ 43.5.
- Refuted if ≥ 45.0.
- Inconclusive if 43.5-45.

# Handoff
Plan agent: combine cycle 270 and cycle 271 carefully. New script `experiments/hybrid_block1_block11_cycle275.py`.
