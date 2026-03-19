# Review (Round 2): Cycle 066 — Weight-space diffusion on topological level sets (Item 45)

Verdict: **APPROVED**

## Round 1 Resolution

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Near-zero acceptance rate (hard rejection on PH) | BLOCKING | **Fixed.** Replaced MCMC rejection with penalized Langevin + calibrated lambda_topo. Continuous topology pressure, no binary accept/reject. |
| 2 | Computational cost underestimate / sequential eps search | MODERATE | **Fixed.** Now 4 ripser calls per gradient step (one per layer), ~2000 per seed. Lr selection is a standard grid, not a nested accept/reject loop. |
| 3 | Missing L1-retrain baseline | MINOR | **Fixed.** Step 3 adds L1-regularized retraining with binary search on lambda to match sparsity. |
| 4 | Track record context | INFO | Acknowledged. Approach now clearly differentiated (soft penalty vs prior hard constraints). |

## New Observations (non-blocking)

### Finite-difference topology gradient direction (MINOR)

The plan says "perturb each layer's weights by eps=1e-4, recompute W1" but does not specify the perturbation direction. If this means adding eps uniformly to all weights in a layer, it computes a directional derivative along the all-ones vector — a crude proxy for the layer's topology sensitivity. This is adequate for a regularizer but should be documented in the script so results are interpretable. If it means perturbing by eps along the current gradient direction, that's better but should be stated explicitly.

Not blocking: the success/refutation criteria will catch whether the heuristic works regardless of this detail.

## Summary

All blocking and moderate issues from round 1 are resolved. The revised plan replaces hard topological rejection with a well-calibrated soft penalty, adds the missing L1-retrain baseline, and has realistic compute estimates. Success and refutation criteria are clear and falsifiable. Approved to run.
