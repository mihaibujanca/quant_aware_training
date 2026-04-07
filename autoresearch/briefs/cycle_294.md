# Item
304 (P2): 1-block [1] K sweep 8000 steps — extend cycle 276.
Test if cycle 276's budget-bounded K={32,64,128,256} curve drops further at 8k.

# Context
- Cycle 276: [1] K∈{32,64,128,256} → {49.65, 48.05, 46.93, 45.39}, all budget-bounded.
- Cycle 287: [1] K=512 4k→8k extension → -1.04 PPL.
- Hypothesis: similar gains at smaller K.

# Dependencies
None.

# Key Files
- experiments/one_block_k_sweep_cycle276.py — direct template, N_STEPS=8000, EVAL_EVERY=200.

# Constraints
- BLOCKS=[1], K_VALUES=[32, 64, 128, 256]
- N_STEPS=8000, EVAL_EVERY=200
- 3 seeds × 4 K = 12 fits.

# Handoff
Plan agent: trivial extension. New script `experiments/one_block_1_k_sweep_8k_cycle294.py`.
