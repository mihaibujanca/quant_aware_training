# Item
294 (P2): Hybrid — block 1 rank-2 + block 11 rank-22 c_proj weight-SVD.
Homogeneous low-rank hybrid. Cycle 275 REFUTED the heterogeneous [1] K=512 + block 11 rank-22 mix.

# Context
- Cycle 280 CONFIRMED: block 1 rank-2 retrain → 49.03 (2,304 params).
- Cycle 270 CONFIRMED: block 11 rank-22 retrain → 54.47 (17,664 params).
- Cycle 275 REFUTED: [1] K=512 + block 11 rank-22 → 48.20 (heterogeneous, worse than [1] alone 43.82).
- Cycle 282 CONFIRMED: block 10 rank-17 → 50.04 (13,824 params), deprecates block 11.
- Open: does homogeneous weight-SVD hybrid compose better?
- PRECONDITION: Cycle 280 CONFIRMED rank-2 at 49.03 (below 55). Precondition met.

# Dependencies
- Cycle 280 rank-2 block 1 SVD infrastructure.
- Cycle 270 rank-22 block 11 SVD infrastructure.

# Key Files
- experiments/block1_lowrank_cycle280.py
- experiments/block11_lowrank_cycle270.py

# Constraints
- Block 1 c_proj replaced with rank-2 SVD factorization (U_2 Σ_2 frozen, B_1=Vh_1[:2,:] trainable = 2×768 + 768 = 2,304).
- Block 11 c_proj replaced with rank-22 SVD (B_11=Vh_11[:22,:] trainable = 22×768 + 768 = 17,664).
- Total trainable ≈ 19,968.
- Joint Adam LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore. 3 seeds. Disjoint splits.
- Confirmed if best_mean ≤ 55.0.
- Refuted if ≥ 58.0.
- Inconclusive otherwise.

# Handoff
Plan agent: combine cycle 280 and cycle 270. New script `experiments/rank_hybrid_1_11_cycle284.py`.
