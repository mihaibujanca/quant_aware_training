# Item
284 (P2): K=64 5-block extended budget (8000 steps) — does cycle 257 plateau?
BLOCKS=[1,2,4,7,8] K=64, 8000 steps, 3 seeds.

# Context
- Cycle 257: K=64 5-block 4000 steps → 62.181, still descending.
- Cycle 254: K=128 5-block 8000 steps → 59.018 (plateaued).
- Open: does K=64 plateau or keep descending?

# Dependencies
None.

# Key Files
- experiments/five_block_k64_cycle257.py — direct template, N_STEPS=8000, EVAL_EVERY=200.

# Constraints
- BLOCKS=[1,2,4,7,8], K=64
- N_STEPS=8000, EVAL_EVERY=200
- 3 seeds.
- Verdict per agenda item 284.

# Handoff
Plan agent: trivial extension. New script `experiments/five_block_k64_8k_cycle274.py`.
