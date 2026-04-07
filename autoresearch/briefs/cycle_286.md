# Item
296 (P0): [2] K=512 1-block — missing data point.
Test if block 2 at K=512 beats cycle 271 [1] K=512 = 43.82.

# Context
- Cycle 271: [1] K=512 → 43.821 (current lowest-PPL Pareto winner).
- Cycle 283: [2] K=16 = 49.46 (beats [1] K=32 by 0.19). Block 2 beats block 1 at every K tested so far.
- Open: does this advantage persist at K=512?

# Dependencies
None.

# Key Files
- experiments/one_block_sweep_cycle271.py — direct template, SUBSETS=[[2]].

# Constraints
- SUBSETS=[[2]]
- K=512
- 3 seeds. Same other config.

# Handoff
Plan agent: trivial BLOCKS swap on cycle 271. New script `experiments/block2_k512_cycle286.py`.
