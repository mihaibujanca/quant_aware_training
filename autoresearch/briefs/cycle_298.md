# Item
308 (P1): SDF rank census + LASER replication on 3D mesh fitting.
Train an MLP on mesh SDF. Measure hidden layer eff_rank and Jacobian rank profile. Apply LASER recipe.

# Context
- Cycle 295/296/297: spirals 2D already rank-1 at init — wrong control.
- SDF: input 3D, surface is 2D manifold. Expected intrinsic dim between 2-3.
- Hypothesis: hidden eff_rank ≈ 3, gradient ensemble eff_rank ≈ 2.

# Dependencies
- aleph/geometric_compression.py (jacobian_rank_profile exists).
- aleph/mesh_datasets.py (load_mesh, sample_sdf_points).

# Key Files
- aleph/mesh_datasets.py
- aleph/geometric_compression.py

# Constraints
- Train an MLP (hidden_dim=256, depth=4) on mesh SDF.
- Per layer: output-covariance eff_rank.
- jacobian_rank_profile for the function's effective dimension.
- LASER: SVD-truncate a layer's weight at r ∈ {2, 3, 5, 10}, retrain B+bias 500 steps, report held-out SDF MSE.
- Verdict per item 308.

# Handoff
Plan agent: design carefully — SDF training is new territory. Check if cycle 295's approach can be adapted. If the infrastructure is too complex for a cycle, consider PLAN INVALID.
