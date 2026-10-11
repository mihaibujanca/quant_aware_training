# Item
264 (P0): 5-block K=128 with extended budget (8000 steps).
Cycle 251 was at step 4000 still descending. Run with 8000 steps to find the actual K=128 5-block ceiling.

# Context
- Cycle 251 (INCONCLUSIVE): 5-block [1,2,4,7,8] K=128 → 59.119 ± 0.009 at step ~3967, monotone descending.
- Cycle 245 (CONFIRMED): same blocks at K=512 → 53.826.
- Open: at extended budget, can K=128 close more of the gap?

# Dependencies
None.

# Key Files
- experiments/five_block_k128_cycle251.py — direct template, just N_STEPS=8000.

# Constraints
- BLOCKS=[1,2,4,7,8], K=128
- N_STEPS=8000, EVAL_EVERY=200 (less frequent to save eval cost over the longer run)
- Other hyperparameters as cycle 251.
- 3 seeds.
- Confirmed if best_mean ≤ 56.0 (significant improvement on cycle 251's 59.119).
- Refuted if best_mean ≥ 58.5 (cycle 251 ceiling already reached).
- Inconclusive if 56-58.5.

# Handoff
Plan agent: trivial N_STEPS swap on cycle 251. New script `experiments/five_block_k128_8k_cycle254.py`.
