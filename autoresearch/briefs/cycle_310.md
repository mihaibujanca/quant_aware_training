# Item
320 (P0): Pairwise LASER composition matrix — all 10 pairs of blocks {1, 2, 3, 10, 11}.

# Context
- Cycle 284 CONFIRMED: [1] rank-2 + [11] rank-22 hybrid → 52.88.
- Cycle 291 REFUTED: [1] rank-2 + [10] rank-5 hybrid → 51.80.
- Cycle 305 REFUTED: 5-block stacking → 73.01.
- Open: which pairs compose non-destructively?

# Dependencies
- Cycle 270/280/282/285/288/289 rank-SVD infrastructure.

# Key Files
- experiments/rank_hybrid_1_11_cycle284.py — 2-block hybrid template.
- experiments/rank_hybrid_1_10_cycle291.py — alternate.

# Constraints
- Optimal ranks: b1 r=2, b2 r=2, b3 r=24, b10 r=5, b11 r=22.
- All C(5, 2) = 10 pairs.
- Each pair: Adam retrain B factors, 3 seeds (or 1 seed for budget), 4000 steps, best-state restore.
- Report 5×5 matrix of best_mean PPL (diagonal = single-block anchor).
- 10 × 3 = 30 fits at 3 seeds is too much. Use SEED=42 only for 10 pairs, extend to 3 seeds if any pair < single-block best.
- Verdict per item 320.

# Handoff
Plan agent: adapt cycle 284/291 templates for the full pairwise sweep. New script `experiments/pairwise_laser_cycle310.py`.
