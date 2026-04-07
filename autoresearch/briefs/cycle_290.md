# Item
300 (P0): [2] K=512 8000 steps — cycle 286 follow-up to settle [2] vs [1] at matched 8k budget.

# Context
- Cycle 286: [2] K=512 4k → 43.56 (still descending, INCONCLUSIVE).
- Cycle 287: [1] K=512 8k → 42.785 (CONFIRMED, -1.04 vs 4k).
- Hypothesis: [2] K=512 8k drops below 42.785?

# Dependencies
None.

# Key Files
- experiments/one_block_1_k512_8k_cycle287.py — direct template, just swap BLOCK=[2].
- experiments/block2_k512_cycle286.py

# Constraints
- BLOCKS=[2], K=512, N_STEPS=8000, EVAL_EVERY=200, 3 seeds, best-state restore. Disjoint splits.

# Handoff
Plan agent: trivial adaptation. New script `experiments/block2_k512_8k_cycle290.py`.
