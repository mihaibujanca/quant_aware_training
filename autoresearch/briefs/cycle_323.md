# Item
336. Post-`mlp_scale` eff_rank census on parameter-golf — does per-channel scaling change the rank picture?

# Context
Cycle 318 measured MLP output eff_rank BEFORE the per-channel `mlp_scale` multiplication (line `x = x + self.mlp_scale[None, None, :] * self.mlp(...)`). The post-scale output is what actually contributes to the residual stream. If post-scale eff_rank shifts, the cycle 318 finding ("b3 is rank-collapsed") needs context-qualification. If post-scale is similar (±20% relative), the pre-scale finding is robust.

Baseline (cycle 318 pre-scale): b0=187.61, b1=160.08, b2=86.05, **b3=4.89**, b4=38.30, b5=39.96, b6=28.21, b7=23.23, b8=30.78, b9=28.85, b10=21.55, b11=13.40, b12=6.94.

# Dependencies
- parameter-golf checkpoint: `/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz` ✓ (exists, 174 MB)
- Template script: `parameter-golf/local_scripts/pgolf_eff_rank_census.py` ✓ (exists, can be copied)
- Parameter-golf venv: `/Users/mihai/Projects/parameter-golf/.venv/bin/python` (MLX, numpy)
- qat autoresearch harness

# Key Files
- `autoresearch/cross_project/parameter_golf.md` — REQUIRED: sections 3 (mlp_scale gotcha) and 6 (worked example, captures pre-scale)
- `parameter-golf/local_scripts/pgolf_eff_rank_census.py` — template for copy-modify
- `autoresearch/results/cycle_318_pgolf_eff_rank.json` — baseline pre-scale profile

# Constraints
- Must capture POST-scale: hook Block.__call__, capture `self.mlp_scale[None, None, :] * self.mlp(...) output`
- Same schema as cycle 318 JSON (eff_rank, rank_95, rank_99, top5_eigvals per block)
- Run cap: ~10 minutes
- Per-block mlp_mult varies (early blocks 1920 hidden, late 1280); use variable loop per cycle 318 pattern

# Handoff
**Plan stage MUST include `cat autoresearch/cross_project/parameter_golf.md` as Step 0 before writing any code.** Exec stage MUST execute Step 0 first.

Write two scripts: qat wrapper `experiments/pgolf_eff_rank_postscale_cycle323.py` + pgolf census `parameter-golf/local_scripts/pgolf_eff_rank_census_post_scale.py`. Save result to `autoresearch/results/cycle_323_pgolf_postscale_eff_rank.json`. Build summary table: pre-scale vs post-scale, highlight blocks shifting ≥50% relative (concentration class break at eff_rank=20 threshold).

**Hypothesis**: post-scale and pre-scale within ±20% relative at every block (rank invariant under linear isomorphism). **Outcomes**: CONFIRMED-rank-invariant (±20% all blocks) → cycle 318 robust; REFUTED-scale-matters (≥50% any block) → rank reshaping; INCONCLUSIVE otherwise.
