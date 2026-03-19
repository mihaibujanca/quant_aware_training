# Cycle 066 Plan — Weight-Space Diffusion on Topological Level Sets (rev 2)

**Item**: 45 (P2 MOONSHOT) | **Script**: `experiments/weight_space_topo_diffusion.py`

## Pre-flight
- P1: Topological level sets in weight space are large enough to traverse toward lower L1 norm.
- P2: Refuted if topology drifts uncontrollably or min-norm matches theta_0.
- P3: No prior cycle tested weight-space diffusion. Cycle 065 used ripser per-layer (works, fast).
- P4: Toy only (spirals, 2->32x4->1). CPU. Ripser on 500 pts in 32D is ~ms per call.
- P5: JSON results + 1-panel bar chart. P6: Single script. P7: CPU, <10 min.

## Design

### Data & Architecture
- Spirals: n=2000, 3 turns, scale=2.0, noise=0.05, seeds={42, 123, 7}.
- Network: 2->32x4->1. Adam lr=1e-3, 2000 epochs, BCE. Require acc >= 90%.

### Step 0: Baseline PH Fingerprint
- Forward 500 stratified training points. Hook post-ReLU at L1-L4.
- Ripser: maxdim=1, thresh=auto. Store PD_0 per layer.
- Record ||theta_0||_1 and baseline accuracy.

### Step 1: Calibrate lambda_topo
- Compute L1_0 = ||theta_0||_1 and W1_0 = mean_over_layers(wasserstein_1(PD_0, PD_random)).
- Set lambda_topo = L1_0 / W1_0 so both penalty terms are comparable at init.

### Step 2: Penalized Langevin (SOFT topology penalty)
- Combined loss: L(theta) = ||theta||_1 + lambda_topo * mean_over_layers(W1(PD_theta, PD_0)).
- 500 gradient steps. lr from {1e-4, 3e-4, 1e-3} (pick first where W1 drift < 20% of W1_0).
- Gradient: sign(theta) for L1 term. Topology term: finite-difference approx per layer,
  perturb each layer's weights by eps=1e-4, recompute W1. (Ripser is non-differentiable.)
- **No hard rejection.** Topology pressure is continuous — prevents drift without binary accept/reject.
- Track: ||theta_t||_1, W1 drift, accuracy at each step. Record theta* = argmin ||theta_t||_1
  subject to acc >= 85% and W1_drift < 30% of W1_0.

### Step 3: Baselines at Matched Sparsity
- Sparsity s = fraction of near-zero weights in theta* (|w| < 1e-3).
- **Magnitude pruning**: prune theta_0 to s sparsity -> acc_mag.
- **L1-regularized retraining**: retrain theta_0 from scratch with L1 penalty lambda
  chosen so final sparsity matches s (binary search on lambda). Same arch, same data,
  same epochs. -> acc_l1. This controls for "retraining finds sparser solutions anyway."
- Report: norm_reduction = 1 - ||theta*||_1 / ||theta_0||_1, acc_topo, acc_mag, acc_l1.

### Step 4: Multi-Seed Validation
- Run Steps 0-3 on all 3 seeds.
- Confirmed: norm_reduction >= 30% AND acc_topo >= max(acc_mag, acc_l1) + 5pp, 2/3 seeds.

## Implementation Notes
- `from ripser import ripser`, `from persim import wasserstein` (both installed).
- Finite-difference topology gradient: perturb one layer at a time (4 layers = 4 ripser calls).
- If W1 drift > 50% of W1_0 at all lr values: REFUTED (topology too sensitive). Stop early.
- Wasserstein on empty diagrams: distance 0 (both empty) or total persistence (one empty).

## Success / Refutation
- **Confirmed**: >= 30% norm reduction + >= 5pp acc advantage over BOTH baselines, 2/3 seeds.
- **Refuted**: W1 drift uncontrollable OR norm reduction < 5% OR no acc advantage over L1 retrain.
- **Inconclusive**: norm reduction achieved but advantage < 5pp over L1 retrain.

## Outputs
- `autoresearch/results/cycle_066_topo_diffusion.json`
- `autoresearch/results/cycle_066_topo_diffusion.png`
