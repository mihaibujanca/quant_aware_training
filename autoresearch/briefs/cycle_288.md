# Item
298 (P1): Block 3 rank-r c_proj weight-SVD — r ∈ {5, 10, 24}.
Block 3 eff_rank = 24.29 (third-lowest after blocks 1/2/10/11).

# Context
- Cycle 278: block 3 c_proj output eff_rank = 24.29.
- Cycle 270: block 11 rank-22 retrain → 54.47.
- Cycle 282: block 10 rank-17 retrain → 50.04.
- Cycle 280: block 1 rank-2 retrain → 49.03 (lowest-param winner).
- Cycle 285: block 2 rank-2 retrain → 49.33 (confirmed but dominated).
- Open: is block 3 a new single-block compression target?

# Dependencies
None.

# Key Files
- experiments/block10_lowrank_cycle282.py — direct template, swap block index.
- experiments/block11_lowrank_cycle270.py.

# Constraints
- For r ∈ {5, 10, 24}: no-retrain PPL + retrain at best r.
- Adapt cycle 282 for block 3.

# Handoff
Plan agent: adapt cycle 282 for block 3. New script `experiments/block3_lowrank_cycle288.py`.
