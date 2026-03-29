# Item
93: Gauss-Newton correction retry with true J^T J — GPT-2 FFN

# Context
Cycle 081 (item 85) tested Newton correction but used **finite-diff full Hessian** (indefinite) instead of true Gauss-Newton approximation **H = J^T J** (PSD by construction). Newton diverged +902% PPL, but steepest descent achieved +14.3% improvement, proving gradient signal is valid.

# Dependencies
- Item 85 (COMPLETED, cycle 081): Identified method failure (wrong Hv approximation, only 10 CG iters, damping ~1e-10)
- Baseline: cycle 059 (500 Adam steps) achieved +18.4% PPL improvement

# Key Files
- `experiments/newton_correction_stacked_gpt2.py`: implement true Gauss-Newton
- `autoresearch/results/cycle_081_newton_correction.json`: baseline results (steepest descent +14.3%, Newton +902%)

# Constraints
- Compute Jacobian J via `torch.func.jacrev` on small batch (256 tokens)
- Form H_gn = J^T J (PSD), solve via CG at **≥100 iterations** (vs 10 in cycle 081)
- Tikhonov damping ~1e-4 (vs negligible ~1e-10 before)
- Blocks [7,5,4,1] at 6×, 3 seeds [42,123,7]
- Single GN step vs 50 Adam steps (baseline: +18.4% PPL)

# Constraints
- Success: single GN step ≤ +12% PPL (within 2× of Adam baseline)
- Refute: diverges or > +25% PPL despite correct PSD Hessian
- Inconclusive: +12–25% PPL (helps but below Adam)

# Handoff
Reimplement with true Gauss-Newton (J^T J via jacrev), proper CG convergence, and strong damping. Steepest descent signal validates gradient, so full Newton should work if Hessian approximation is correct. This is the final attempt before shelving analytic correction.
