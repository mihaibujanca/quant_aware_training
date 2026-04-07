# Item
303 (P1): Block 0 rank-r c_proj weight-SVD — r ∈ {10, 30, 72}.
Block 0 eff_rank = 71.93, untested.

# Context
- Cycle 278: block 0 eff_rank = 71.93 (moderate, between rank-concentrated blocks 1-4 and high-rank 5-9).
- Cycle 198: block 0 is irreplaceable (block 0 input rank is 269, others ~1 — block 0 transforms embedding diversity).
- Cycle 187: dropping block 0 breaks the model (PPL 1923).
- Open: can block 0 c_proj be compressed via weight-SVD?

# Dependencies
None.

# Key Files
- experiments/block3_lowrank_cycle288.py — direct template, swap block index.

# Constraints
- Block 0 c_proj rank-r for r ∈ {10, 30, 72}. No-retrain sweep + retrain at r=72 (matching eff_rank).
- 3 seeds retrain, 4000 steps.

# Handoff
Plan agent: adapt cycle 288 for block 0. New script `experiments/block0_lowrank_cycle293.py`.
