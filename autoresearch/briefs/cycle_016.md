# Cycle 016 Brief

## Item
**32. GELU → 3-piece PWL approximation fidelity — toy GELU MLP**

## Context
Cycles 009–015 tested single-affine collapse, affine map clustering, spectral graph coarsening, and manifold intrinsic dimension on both toy ReLU and GPT-2 GELU networks. All failed as compression signals. The core impediment: **GELU smoothness may make piecewise-affine theory inapplicable to transformers.** This item directly tests that assumption by replacing GELU with low-piece PWL and measuring accuracy drop.

## Dependency
None. Independent feasibility study.

## Key Files
- **Experiment**: `experiments/gelu_pwl_approximation.py` (create)
- **Config**: Spirals GELU MLP (2→100D embedding→4×64→1), same depth as cycle 010 (cycle_010_single_affine.py reference)
- **Reference network**: Train fresh GELU network to convergence on spirals
- **PWL approximation**: g̃(x) = {0 for x<−3, linear ramp [−3,3] matching GELU at ±3 and 0, x for x>3}

## Constraints
- **No retraining** after PWL substitution (oracle approximation fidelity test)
- **Sweep piece count**: 1, 3, 5, 10 → find minimum where ReLU theory kicks in
- **Accuracy threshold**: 3-piece < 3% drop → **confirmed**; 10-piece < 1% drop → refined confirmation

## Handoff
Script must:
1. Train GELU MLP to convergence on spirals (use spirals trainer from cycle 010)
2. Save trained weights
3. For each piece count in {1,3,5,10}, substitute PWL, evaluate test accuracy, log
4. Plot accuracy vs piece count with 3% and 1% thresholds marked
5. Report pass/fail/inconclusive per agenda criterion
6. Outcome feeds item 33 (GPT-2 analog): if refuted (>5% drop at 10-piece), skip 33
