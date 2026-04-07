# Item
305 (P0): Effective rank census on trained spirals/moons MLPs.
Train 32×12 MLP on spirals and moons. Capture post-ReLU activations per layer. SVD output covariance → eff_rank. Compare to random init.

# Context
- Cycle 278: GPT-2 c_proj output eff_rank profile {block 1: 1.94, block 2: 1.05, block 5: 179.19, block 11: 22.49}.
- Open: is the collapse transformer-specific or generic to overparameterized training on low-D tasks?
- Hanin & Rolnick: activation patterns bounded by N^d where d=input dim. For d=2 predicts very few effective dimensions.

# Dependencies
None.

# Key Files
- aleph/datasets.py → make_spirals, make_moons
- aleph/__init__.py → make_mlp

# Constraints
- hidden_dim=32, depth=12, input_dim=2, output_dim=1
- Spirals: n_samples=2000, noise=0.5, n_turns=3
- Moons: n_samples=2000, noise=0.1
- Train with Adam lr=0.001 until ≥95% train acc
- 3 seeds (42, 137, 2024)
- Per hidden layer ℓ ∈ {0..11}: capture post-ReLU Y_ℓ ∈ R^(N, 32). Compute C_ℓ = Y_ℓ^T Y_ℓ / N, SVD, eff_rank = exp(-Σ p_i log p_i).
- Report per-layer eff_rank at init AND post-training.
- Confirmed if post-training eff_rank ≤ 3 in ≥8 of 12 layers AND init eff_rank ≥ 20 in ≥8 of 12 layers.
- Refuted if post-training eff_rank ≥ 10 in ≥6 layers.

# Handoff
Plan agent: new script `experiments/spirals_moons_eff_rank_cycle295.py`. Uses aleph.datasets and aleph.make_mlp.
