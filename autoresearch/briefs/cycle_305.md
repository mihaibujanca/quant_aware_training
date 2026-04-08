# Item
315 (P0): Multi-block LASER stacking — blocks [1, 2, 3, 10, 11] each at optimal rank, all compressed simultaneously.

# Context
- Cycle 280 block 1 rank-2 → 49.03 (2,304 params).
- Cycle 285 block 2 rank-2 → 49.33 (2,304 params).
- Cycle 288 block 3 rank-24 → 50.59 (19,200 params).
- Cycle 289 block 10 rank-5 → 51.47 (4,608 params).
- Cycle 270 block 11 rank-22 → 54.47 (17,664 params, later deprecated by cycle 282).
- Cycle 275 REFUTED: [1] K=512 + block 11 rank-22 heterogeneous hybrid (48.20).
- Cycle 284 CONFIRMED: block 1 rank-2 + block 11 rank-22 homogeneous hybrid (52.88).
- Cycle 291 REFUTED: block 1 rank-2 + block 10 rank-5 hybrid (51.80).
- Open: does stacking ALL 5 concentrator blocks at their optimal rank work?

# Dependencies
- All cycle 270/280/285/288/289 rank-SVD infrastructure.

# Key Files
- experiments/rank_hybrid_1_11_cycle284.py — 2-block hybrid template.
- experiments/rank_hybrid_1_10_cycle291.py.

# Constraints
- BLOCKS = [1, 2, 3, 10, 11] each at its best rank from prior cycles:
  - block 1 rank-2
  - block 2 rank-2
  - block 3 rank-24
  - block 10 rank-5
  - block 11 rank-22
- All 5 c_proj replaced simultaneously with rank-r factorizations.
- Joint Adam retrain B factors (trainable = 2+2+24+5+22 = 55 rank-total × 768 per factor + biases ≈ 42K params).
- 3 seeds, 4000 steps, best-state restore, disjoint splits.
- Confirmed if best_mean ≤ 50.
- Refuted if best_mean ≥ 56.

# Handoff
Plan agent: combine cycles 270/280/285/288/289 into a 5-block hybrid. New script `experiments/multi_block_laser_cycle305.py`.
