# Item
297 (P0): [1] K=512 extended budget (8000 steps).
Can Adam push cycle 271's 43.82 floor below 43?

# Context
- Cycle 271: [1] K=512 4000 steps → 43.821.
- Cycle 286: [2] K=512 4000 steps → 43.563 (still descending, also not at cap).
- Cycle 254 (K=128 5-block 8k): plateaued.
- Open: is [1] K=512 budget-limited?

# Dependencies
None.

# Key Files
- experiments/one_block_sweep_cycle271.py — direct template, just N_STEPS=8000 + EVAL_EVERY=200.

# Constraints
- BLOCKS=[1] K=512
- N_STEPS=8000, EVAL_EVERY=200
- 3 seeds, best-state restore, disjoint splits.

# Handoff
Plan agent: trivial extension. New script `experiments/one_block_1_k512_8k_cycle287.py`.
