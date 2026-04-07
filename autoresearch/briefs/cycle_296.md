# Item
306 (P0): LASER recipe on spirals — does cycle 280 replicate on a known-2D task?

# Context
- Cycle 295: trained 32×12 spirals and moons MLPs, eff_rank collapses to ~1 at most layers (2D input).
- Cycle 280: GPT-2 block 1 rank-2 c_proj retrain → 49.03 (-4.86 vs float).
- Cycle 292: GPT-2 block 1 rank-1 retrain → 50.08 (+0.51 vs rank-2, same direction).
- Open: does rank-2 weight SVD retrain match float on a provably 2D task?

# Dependencies
- Cycle 295 trained models (or retrain).

# Key Files
- experiments/spirals_moons_eff_rank_cycle295.py — trained models (if saved; else retrain inline).
- experiments/block1_lowrank_cycle280.py — SVD + retrain template.

# Constraints
- From cycle 295's trained spirals net (or retrain), pick the layer with smallest measured eff_rank.
- SVD W at that layer. Sweep r ∈ {1, 2, 3, 5}.
- For each r: no-retrain held-out acc; Adam fine-tune B + bias 500 steps, report best held-out acc.
- Repeat for moons.
- 3 seeds.
- Confirmed if rank-2 retrain held-out acc ≥ float − 0.5% on spirals AND no-retrain rank-2 ≥ 80% of float.
- Refuted if rank-2 retrain held-out gap > 5%.

# Handoff
Plan agent: design via cycle 295 training infrastructure + cycle 280 SVD retrain. New script `experiments/spirals_moons_laser_cycle296.py`.
