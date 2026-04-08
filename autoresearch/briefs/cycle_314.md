# Item
324 (P0): Rank-1 pairwise LASER composition matrix — all 10 pairs of {1, 2, 3, 10, 11}, each block at rank-1.

# Context
- Cycle 311 CONFIRMED: all 5 concentrator blocks work at rank-1 (1,536 params each).
- Cycle 310 pairwise matrix used OPTIMAL ranks (b1 r=2, b11 r=22). Most pairs non-destructive; only (b10, b11) broke.
- Open: at rank-1 per block, how does the composition matrix look?

# Dependencies
None.

# Key Files
- experiments/pairwise_laser_cycle310.py — template.
- experiments/rank1_all_concentrators_cycle311.py — rank-1 infrastructure.

# Constraints
- 10 pairs at rank-1 per block.
- Trainable per pair: 2 × 1,536 = 3,072 params.
- 1 seed (SEED=42) per pair for the matrix; extend to 3 seeds if any pair dominates.
- 4000 steps, best-state restore.

# Handoff
Plan agent: adapt cycle 310 template with all blocks at rank-1. New script `experiments/rank1_pairwise_laser_cycle314.py`.
