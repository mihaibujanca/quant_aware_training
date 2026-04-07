# Item
282 (P0): K=256 4-block sweep — fill the missing data point.
SUBSETS=[[1,2,4,7], [1,4,7,8], [1,2,7,8]] at K=256.

# Context
- K=512 4-block (cycle 255): [1,2,4,7]=49.609.
- K=128 4-block (cycle 265): [1,2,4,7]=53.647.
- K=256 3-block (cycle 223): 47.509.
- K=256 6-block (cycle 256): 64.181.
- Predicted K=256 4-block ≈ 51.

# Dependencies
None.

# Key Files
- experiments/four_block_k128_sweep_cycle265.py — direct template, K swap.

# Constraints
- SUBSETS=[[1,2,4,7], [1,4,7,8], [1,2,7,8]]
- K=256
- 3 seeds × 3 subsets = 9 fits.
- Same other config.

# Handoff
Plan agent: trivial K swap on cycle 265. New script `experiments/four_block_k256_sweep_cycle272.py`.
