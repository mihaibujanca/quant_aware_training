# Item
319 (P0): Δy-aligned rank-r LASER at block 8 — use Vh_y (not Vh_w) as the subspace basis.

# Context
- Cycle 306: at block 8, Δy is rank-1 dominated on its OWN SVD (top-1 energy 91%), but 53% of that lives in a direction misaligned with W_proj top-2.
- Cycle 280 recipe uses Vh_w (W_proj SVD) — which catches only 47% of block 8's curvature.
- Open: does a Δy-aligned LASER capture block 8's signal better?

# Dependencies
None.

# Key Files
- experiments/block8_deep_dive_cycle306.py — has Vh_y computation.
- experiments/block11_lowrank_cycle270.py / block10_lowrank_cycle282.py — retrain template.

# Constraints
- At block 8: compute Δy on calib. SVD Δy → Vh_y (top-2 or top-5 right singular vectors).
- Build a rank-r c_proj replacement that preserves Δy's top-r directions (not W_proj's).
- One approach: use Vh_y as the target output subspace. Find the rank-r c_proj that projects onto span(Vh_y).
- More concrete: W_proj_new = W_proj @ (Vh_y^T @ Vh_y)  (project W_proj's columns onto Δy's top-r subspace).
- Eval PPL no-retrain + retrain at r=2.
- Verdict per item 319.

# Handoff
Plan agent: design carefully. New script `experiments/block8_dy_aligned_laser_cycle309.py`.
