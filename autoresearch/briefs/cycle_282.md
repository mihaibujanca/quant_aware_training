# Item
292 (P0): Block 10 rank-17 weight-SVD — extend cycle 270 to second-lowest eff_rank block.

# Context
- Cycle 270: block 11 rank-22 weight-SVD + retrain → 54.47 (17,664 params).
- Cycle 278: block 10 eff_rank = 16.70 (second-lowest after blocks 1/2).
- Cycle 280: block 1 rank-2 retrain → 49.03 (2,304 params, new Pareto anchor).
- Open: does block 10 work similarly to block 11?

# Dependencies
None.

# Key Files
- experiments/block11_lowrank_cycle270.py — direct template, swap block index.

# Constraints
- Adapt cycle 270 for block 10. r ∈ {5, 17, 50}.
- For r=17: Adam retrain (trainable = 17×768 + 768 = 13,824 params), 3 seeds.
- Same hyperparameters.

# Handoff
Plan agent: adapt cycle 270 for block 10. New script `experiments/block10_lowrank_cycle282.py`.
