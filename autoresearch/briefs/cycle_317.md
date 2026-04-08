# Item
327 (P1): 3-block LASER composition — 10 triples from {1, 2, 3, 10, 11}.

# Context
- Cycle 310 pairs: 9/10 non-destructive, (b10, b11) breaks.
- Cycle 305: 5-block stack = 73 PPL (destruction).
- Cycle 316: (b10, b11) destruction is interactive, both must be present.
- Open: at triples (between 2 and 5), where does cascade destruction start?

# Dependencies
None.

# Key Files
- experiments/pairwise_laser_cycle310.py — 2-block template.
- experiments/multi_block_laser_cycle305.py — 5-block template.

# Constraints
- Optimal ranks per block: b1 r=2, b2 r=2, b3 r=24, b10 r=5, b11 r=22.
- 10 triples from C(5, 3).
- Per triple: joint Adam, 1 seed (42), 4000 steps, best-state restore.
- Verdict per item 327.

# Handoff
Plan agent: adapt cycle 310 for triples. New script `experiments/triples_laser_cycle317.py`.
