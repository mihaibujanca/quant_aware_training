# Item
286 (P0): 1-block [1] K sweep — K ∈ {32, 64, 128, 256}. Find the 1-block floor.

# Context
- Cycle 271: [1] K=512 → 43.82.
- Cycle 223/225/230: K sweeps at 3 blocks showed K=1 still worked (cycle 230).
- Open: at 1 block, how low can K go?

# Dependencies
None.

# Key Files
- experiments/one_block_sweep_cycle271.py — template.

# Constraints
- SUBSETS=[[1]]
- K_VALUES=[32, 64, 128, 256]
- 3 seeds × 4 K values = 12 fits.
- Same other config.
- Verdict per item 286.

# Handoff
Plan agent: adapt cycle 271 template for K sweep. New script `experiments/one_block_k_sweep_cycle276.py`.
