# Item
307 (P0): Effective rank trajectory through training — when does the collapse happen?

# Context
- Cycle 295 already showed init eff_rank ≤ 1.0 by layer 7 for both spirals and moons. The "init-rank" CONFIRMED case is already the expected outcome.
- Item 307 captures the full training trajectory (eff_rank per layer at every 100 steps).
- Expected verdict: CONFIRMED init-rank (random init already at rank-1, no collapse needed).

# Dependencies
- Cycle 295/296 infrastructure.

# Key Files
- experiments/spirals_moons_eff_rank_cycle295.py — training loop to instrument.

# Constraints
- Train fresh spirals 32×12 net, record eff_rank per layer at every 100 training steps.
- Also record train loss and held-out accuracy.
- 3 seeds.
- Plot per-layer eff_rank trajectory.
- Identify gradual vs phase-transition vs init-rank.

# Handoff
Plan agent: extend cycle 295 training to record eff_rank snapshots. New script `experiments/eff_rank_trajectory_cycle297.py`.
