# Item
299 (P2): Block 10 rank-5 c_proj — sub-eff_rank stress test of cycle 282.
Can block 10 at rank-5 (much below its eff_rank=16.70) still beat float?

# Context
- Cycle 282: block 10 rank-17 retrain → 50.04 (13,824 params, Pareto).
- Cycle 278: block 10 eff_rank = 16.70.
- Cycle 280 analog: block 1 rank-2 (below eff_rank 1.94) worked at 49.03.
- Open: can rank-r < eff_rank still work for block 10?

# Dependencies
None.

# Key Files
- experiments/block10_lowrank_cycle282.py — direct template, swap r.

# Constraints
- Block 10 c_proj rank-5 SVD. Retrain B factor (5, 768) + bias (768) = 4,608 params.
- 3 seeds, 4000 steps, best-state restore, disjoint splits.

# Handoff
Plan agent: adapt cycle 282 for rank-5. New script `experiments/block10_rank5_cycle289.py`.
