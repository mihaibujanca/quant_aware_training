# Audit: Cycle 066 — Weight-space diffusion on topological level sets (Item 45)

**Auditor**: Claude (audit agent)
**Date**: 2026-03-19
**Verdict reviewed**: REFUTED
**Audit result**: REFUTED stands, but flagged as METHOD FAILURE, not hypothesis refutation

---

## G1: Implementation — Was Langevin correctly implemented?

**FLAG: Multiple implementation issues**

1. **Not Langevin**: No noise term `sqrt(2*lr) * N(0,1)`. This is proximal gradient descent, not Langevin dynamics. The name is misleading.

2. **No task loss**: `L(theta) = ||theta||_1 + lam * W1` has no BCE term. The L1 subgradient `sign(theta)` pushes every parameter toward zero at rate `lr` per step. Without a task-loss gradient providing a restoring force, accuracy collapses on step 1 (~45-50%), and the topology penalty is too weak to recover it. This is the root cause of the failure.

3. **Scalar topology gradient per layer**: `langevin_step` computes one directional derivative along the all-ones vector per layer (4 ripser calls per step), then broadcasts that scalar as the gradient for every parameter in the layer (line 210: `lin.weight.data -= lr * lam * dw1`). This is not a gradient — it is a single scalar applied uniformly. The actual per-parameter topology gradient would require ~3297 perturbations per step. The "topology pressure" is essentially a uniform additive constant per layer, which cannot preserve fine-grained boundary structure.

4. **L1 applied to all params, topology only to hidden**: Output layer gets pure L1 shrinkage with no topology counterforce (lines 206-211). Output layer weights control decision boundary scale directly — shrinking them unchecked destroys classification margin.

5. **Theta-star sparsity mismatch**: Baselines use sparsity measured from the final Langevin state (line 370-378, a second 300-step run), but `acc_topo` is measured at the best-checkpoint from the history. Sparsity at the best checkpoint may differ from final-state sparsity. At ~1-2% sparsity this gap is negligible, but the logic is unsound.

## G2: Seeds

3 seeds (42, 123, 7). All completed. Training accuracy 99.7-99.9% (above 88% threshold). Criterion requires 2/3 confirmed. 0/3 met threshold. Seed coverage is adequate for a refutation of this specific algorithm.

## G3: Results interpretation

| Seed | norm_red | acc_topo | acc_mag | acc_l1 | adv_mag | adv_l1 |
|------|----------|----------|---------|--------|---------|--------|
| 42   | 7.6%     | 83.5%    | 99.9%   | 98.3%  | -16.5pp | -14.8pp |
| 123  | 5.4%     | 83.0%    | 99.8%   | 99.1%  | -16.8pp | -16.1pp |
| 7    | 3.8%     | 90.8%    | 99.7%   | 97.0%  | -8.9pp  | -6.3pp  |

Criterion (>=30% norm_red AND >=5pp over both baselines): **not met for any seed**. Norm reduction is 5-13x below threshold. acc_topo is 6-17pp *below* baselines, not above. The numbers unambiguously fail the criterion.

## G5: Is this a genuine refutation or a method failure?

**METHOD FAILURE.** The refutation is valid for the specific algorithm tested (L1 + W1 penalty without task loss), but does not refute the broader hypothesis that topology-aware weight-space navigation could enable better compression. Specific reasons:

- Removing the task loss from the objective makes the experiment test an unreasonably weak algorithm. Standard practice (e.g., lottery ticket, L1-retrain) always includes task loss.
- The topology gradient approximation (1 scalar per layer) is too crude to provide meaningful geometric guidance.
- Without noise, this is not Langevin — the stochastic exploration that would justify the "diffusion on level sets" framing is absent.
- The baselines (magnitude pruning, L1-retrain) both include task information by construction. Comparing a task-free method against task-aware baselines is an unfair comparison that guarantees failure.

A fair test would need: `L(theta) = BCE(model(X), y) + alpha * ||theta||_1 + lam * W1(PD_theta, PD_0)` with per-parameter topology gradients (or at least per-neuron), plus an actual noise term if claiming Langevin.

---

## Summary

- **Verdict**: REFUTED stands (the algorithm as implemented fails decisively)
- **Guardrail flags**: G1 (implementation issues), G5 (method failure, not hypothesis refutation)
- **Key issue**: No task loss in objective guarantees accuracy collapse; topology gradient is a single scalar per layer, too crude to guide compression
- **Recommendations**:
  1. Do NOT mark the hypothesis "topology-guided compression" as dead based on this result
  2. If revisiting: add BCE task loss to objective, use per-neuron finite differences (or backprop through a differentiable PH proxy), add noise term
  3. Consider whether the 4-ripser-calls-per-step budget is fundamentally too low for meaningful topology gradients in 3297-parameter space
