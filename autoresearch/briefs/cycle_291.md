# Item
301 (P0): Rank hybrid [1] rank-2 + [10] rank-5 — homogeneous low-rank weight-SVD hybrid.

# Context
- Cycle 280: block 1 rank-2 retrain → 49.03 (2,304 params).
- Cycle 289: block 10 rank-5 retrain → 51.47 (4,608 params).
- Cycle 284: block 1 rank-2 + block 11 rank-22 hybrid → 52.88 (asymmetric — block 11 benefits from block 1, not vice versa).
- Hypothesis: homogeneous low-rank weight-SVD hybrid might compose better than mixed K=512+SVD.

# Dependencies
- Cycle 280 block 1 rank-2 infrastructure.
- Cycle 289 block 10 rank-5 infrastructure.

# Key Files
- experiments/rank_hybrid_1_11_cycle284.py — direct template, swap block 11 for block 10.

# Constraints
- Block 1 c_proj rank-2 SVD + block 10 c_proj rank-5 SVD. Joint Adam retrain both trainable factors.
- Total trainable ≈ 2,304 + 4,608 = 6,912.
- 3 seeds, 4000 steps, best-state restore, disjoint splits.

# Handoff
Plan agent: adapt cycle 284. New script `experiments/rank_hybrid_1_10_cycle291.py`.
