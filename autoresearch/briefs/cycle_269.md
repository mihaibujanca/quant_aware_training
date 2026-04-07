# Item
279 (P2): K=1 6-block extended budget (16000 steps) — settle whether cycle 240's catastrophe is budget-limited.
BLOCKS=[1,2,4,5,7,8] K=1 Adam W_out+b_out, 16000 steps, 3 seeds.

# Context
- Cycle 240 (REFUTED): K=1 6-block at 4000 steps → 96.886 ± 0.071. All seeds best at step 4000 (still descending).
- Cycle 254 (5-block K=128 8000 steps): plateaued and overfit. So 16k may or may not converge.
- This cycle settles whether cycle 240 was budget-limited.

# Dependencies
None.

# Key Files
- experiments/adam_6block_k1_cycle240.py — direct template, just N_STEPS=16000.

# Constraints
- BLOCKS=[1,2,4,5,7,8], K=1
- N_STEPS=16000, EVAL_EVERY=400 (less frequent for the longer run)
- Other hyperparameters as cycle 240.
- 3 seeds.
- Verdict per agenda item 279 (verify in agenda).

# Handoff
Plan agent: trivial extension on cycle 240. New script `experiments/k1_6block_16k_cycle269.py`.
