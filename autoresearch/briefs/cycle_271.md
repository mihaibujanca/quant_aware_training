# Item
281 (P0): K=512 1-block compression — where does the cascade end?
Test [1], [4], [7] at K=512.

# Context
- Cycle 266: 2-block [1,4] K=512 → 43.85.
- Cycle 210: 3-block [1,4,7] → 46.56.
- Predicted 1-block ≈ 41-42 PPL based on the concave cascade curve.

# Dependencies
None.

# Key Files
- experiments/two_block_sweep_cycle266.py — direct template, just SUBSETS=[[1],[4],[7]].

# Constraints
- SUBSETS=[[1], [4], [7]]
- K=512
- 3 seeds × 3 subsets = 9 fits.
- Same other config as cycle 266.
- Verdict per agenda item 281 (verify).

# Handoff
Plan agent: trivial SUBSETS swap on cycle 266. New script `experiments/one_block_sweep_cycle271.py`.
