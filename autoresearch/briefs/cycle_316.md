# Item
326 (P0): (b10, b11) destructive pair deep-dive — mechanistic attribution.
Test 3 candidate mechanisms: (M1) residual stream redundancy, (M2) LN coupling, (M3) optimization pathology.

# Context
- Cycle 310: (b10, b11) is the unique destructive pair at +6.25 PPL. 9/10 other pairs fine.
- Open: why specifically this pair?

# Dependencies
None.

# Key Files
- experiments/pairwise_laser_cycle310.py — joint pair infrastructure.
- experiments/block11_lowrank_cycle270.py / block10_lowrank_cycle282.py — single-block.
- experiments/curvature_residual_geometric.py — Δy computation.

# Constraints
Per agenda item 326:
- M1 test: cos angle between Δy_10 top SVs and Δy_11 top SVs via cycle 302 framework.
- M2 test: sequential Adam (fit b10 first, then freeze and fit b11) vs joint Adam.
- M3 test: one-sided restore (b10 at rank-5 with b11 float, and vice versa).
- 1 seed (42), 2000 steps per phase.

# Handoff
Plan agent: new script `experiments/b10_b11_deep_dive_cycle316.py`.
