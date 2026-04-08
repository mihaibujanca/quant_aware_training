# Item
323 (P2): Pure no-retrain LASER baseline across all 12 blocks.
Compute 12×7 matrix of no-retrain PPL for ranks {1, 2, 4, 8, 16, 32, 64}. Compare uncentered vs calibrated (SVD-LLM weighted).

# Context
- Cycle 280: block 1 rank-2 no-retrain init = 57.66; retrain = 49.03. Gap = 8.6 PPL.
- LASER/SVD-LLM papers claim no-training works.
- Open: is our init suboptimal or is Adam gap genuinely from non-SVD optimum?

# Dependencies
None.

# Key Files
- experiments/block1_lowrank_cycle280.py — template.
- experiments/block11_lowrank_cycle270.py — template.

# Constraints
- For block l ∈ {0..11}, rank r ∈ {1, 2, 4, 8, 16, 32, 64}: 84 cells total.
- No retrain: just replace c_proj.weight with U[:, :r] @ diag(S[:r]) @ Vh[:r, :] and measure PPL.
- Also compute calibrated variant: weighted SVD using E[xx^T] from calibration.
- Cross-check against cycle 280/285/288/289/292 published no-retrain numbers.
- Wall ~17 min.

# Handoff
Plan agent: design carefully. New script `experiments/no_retrain_laser_baseline_cycle313.py`.
