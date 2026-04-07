# Item
293 (P1): 1-block K sweep for block 2 — K ∈ {16, 32, 64}.
Block 2 is the lowest-eff_rank block (1.05). Test if K compression can also work cheaply on it.

# Context
- Cycle 276: block 1 K sweep {32, 64, 128, 256} → confirmed.
- Cycle 278: block 2 eff_rank = 1.05 (essentially rank-1).
- Cycle 281: [1,2] K=512 2-block → 44.34 (worse than [1] alone at 43.82).
- Open: does block 2 alone work well at small K?

# Dependencies
None.

# Key Files
- experiments/one_block_k_sweep_cycle276.py — direct template, change BLOCKS=[[2]] and K_VALUES=[16, 32, 64].

# Constraints
- BLOCKS=[2]
- K_VALUES=[16, 32, 64]
- 3 seeds × 3 K = 9 fits.
- Same other config.

# Handoff
Plan agent: trivial swap. New script `experiments/block2_k_sweep_cycle283.py`.
