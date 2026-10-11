# Item
255 (P0): Drop block 5 from 6-block compression — try [1,2,4,7,8] (5 blocks).
Test whether dropping cycle 243's biggest single-block contributor (block 5, +5.10 ΔPPL) actually beats the 6-block 59.5 floor.

# Context
- Cycle 221: 6-block [1,2,4,5,7,8] joint Adam → 59.515 ± 0.094.
- Cycle 243: per-block restoration showed block 5 has the largest ΔPPL when restored to float (+5.10).
- Cycle 244 (just-completed): in fresh-fit, mid-blocks are NOT dominant — depth tracks deepest block. So the cycle 243 finding may not predict cycle 245.

# Dependencies
None.

# Key Files
- experiments/adam_6block_earlystop_cycle221.py — direct template, change BLOCKS to [1,2,4,7,8].

# Constraints
- BLOCKS=[1,2,4,7,8], K=512.
- Same hyperparameters as cycle 221.
- 3 seeds (42, 137, 2024).
- Confirmed if best_mean ≤ 54.0 (block 5 drop closes most of the gap).
- Refuted if best_mean ≥ 57.0 (cascade-joint, no single-block remediation).
- Inconclusive if 54-57.

# Handoff
Plan agent: copy cycle 221 with BLOCKS swap. New script `experiments/drop_block5_cycle245.py`. Output JSON + summary, comparison to cycle 221 (59.515) and cycle 210 (46.556).
