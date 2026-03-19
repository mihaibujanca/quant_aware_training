# Cycle 066: Item 45 — Weight-space diffusion on topological level sets

**Item**: 45 (P2 Moonshot)
**Task**: Explore weight-space constrained by topological invariance.
**Hypothesis**: Networks are over-parameterized for their topological complexity; topological level sets in weight space are large and can be traversed toward minimum L1 norm.

## Method
- Baseline: persistent diagram (PD₀) of layer-wise representations on training data using Ripser.
- Projected Langevin: θ_{t+1} = θ_t − ε·∇||θ||₁ + √(2ε)·noise (L1 sparsity drift).
- Rejection criterion: if Wasserstein₁(PD_t, PD₀) > δ, reject step (topology changed).
- Endpoint: minimum-norm point on trajectory (θ*).
- Pruning: zero out small weights in θ*, measure accuracy.
- Baselines: standard magnitude pruning of θ₀, L1-regularized retraining.

## Success Criteria
- **Confirmed**: ≥30% weight norm reduction while preserving PD within δ AND pruned θ* accuracy ≥5pp better than standard magnitude pruning at matched sparsity.
- **Refuted**: topological level set is tiny (any perturbation changes PD) OR minimum-norm point has essentially same norm as θ₀.
- **Inconclusive**: norm reduction achieved but pruned accuracy not better than baseline.

## Network Setup
- Toy ReLU: 2D input, 2→32×4→1 on spirals (acc ≈98.8%).
- 3 seeds each for reproducibility.

## Status
Ready for implementation. Depends on: none.
