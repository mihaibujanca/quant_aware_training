# Item
276 (P0): 2-block K=512 sweep — [1,4], [1,7], [4,7].
Where does the curve go below 3-block?

# Context
- Cycle 210: 3-block [1,4,7] K=512 → 46.556.
- Cycle 255: 4-block [1,2,4,7] K=512 → 49.609.
- Count-driven story: each block dropped saves ~3-6 PPL. Predicted 2-block ≈ 43-44 PPL.

# Dependencies
None.

# Key Files
- experiments/four_block_sweep_cycle255.py — direct template, swap SUBSETS.

# Constraints
- SUBSETS=[[1,4], [1,7], [4,7]]
- K=512
- 3 seeds, 4000 steps, EVAL_EVERY=100, best-state restore. Disjoint splits.
- Verdict per agenda item 276 (verify in agenda).

# Handoff
Plan agent: trivial SUBSETS swap on cycle 255 template. New script `experiments/two_block_sweep_cycle266.py`.
