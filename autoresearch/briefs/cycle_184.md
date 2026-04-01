# Cycle 184 — Residual Stream Trajectory Divergence: GELU vs ReLU+bias

## Item
197: **Residual stream trajectory divergence—GELU vs ReLU+bias on GPT-2** [INTERPRETABILITY]

## Context
Cycles 181–182 showed ReLU+bias swap fails at +63–88% PPL despite correct per-block biases (cycle 182: iterative 37.0 vs precomputed 42.6, +5.6 PPL improvement insufficient). Distribution shift is real but not dominant error source. Remaining damage must come from GELU vs ReLU **shape difference** near z=0, not bias calibration alone.

## Goal
Measure HOW and WHERE residual stream diverges across 12 blocks when GELU→ReLU+bias. Quantify divergence growth rate, rank, and concentration to test whether activation shape mismatch explains the +63% PPL failure.

## Key Measurements (2048 OWT tokens, 3 models)
1. **Per-layer divergence**: ||r_gelu_l − r_relu+bias_l|| / ||r_gelu_l|| at each block (12 points)
2. **Divergence rank**: SVD of [r_gelu − r_relu+bias] across tokens — concentrated or diffuse?
3. **Growth rate**: Fit ||D_l|| vs layer index. Exponential (R²>0.8) or linear?
4. **Transition zone**: Top-50 divergent tokens/layer—what % have |z|<1 (GELU vs ReLU differ)?
5. **α-interpolation**: GELU_α = α·GELU + (1−α)·ReLU with iterative bias, α∈{0,0.25,0.5,0.75,1}. Smooth or sharp?

## Success Criteria
- **CONFIRMED**: Exponential growth (R²>0.8) + rank≤32 + smooth α curve
- **REFUTED**: Diffuse rank>256 or sharp α transition
- **INCONCLUSIVE**: Linear growth or rank varies wildly

## Handoff
Focus on structure (rank, growth, concentration), not magnitude alone. If α-smooth, shape mismatch explains failure. If sharp threshold, phase transition or interaction dominates. Compute ~15 min.
