# Audit — Cycle 064: Topology-preserving distillation (Item 42)

**Status: REFUTED — but due to implementation bugs (MP5/MP6), not genuine refutation of the hypothesis.**

## G1: Distance-matrix loss implementation

**FAIL — numerical bug invalidates all dmat results.**

The `distance_matrix_loss` function (line 160-175) computes `d2.clamp(min=0).sqrt()`. The gradient of `sqrt(x)` at `x=0` is `1/(2*sqrt(0)) = inf`. Post-ReLU activations frequently have zero pairwise distances (dead neurons produce identical zero rows). This causes NaN gradients.

Evidence from the JSON results: **every single dmat run** (24/24: 3 seeds x 4 lambdas x 2 datasets) goes NaN at epoch 100 (the first logged checkpoint after epoch 0). Training was dead from epoch ~1-99 onward. The 0.485-0.527 accuracies are not dmat results — they are random-init predictions from a model that received NaN gradients for 95%+ of training.

Fix: `d2.clamp(min=1e-12).sqrt()`, or use squared distances (drop the `.sqrt()` entirely — Frobenius on squared distances avoids the singularity and is equally valid for topology matching).

The CKA and activation-MSE implementations appear correct (no NaN in any run), and their results are plausible (neutral vs output-only).

## G2: Seeds

3 seeds (42, 123, 7) x 2 datasets. Adequate for a toy experiment. However, since dmat is NaN in all runs, seed variation is irrelevant — it's testing random initialization, not the method.

## G3: Is -0.05pp genuine or MP5/MP6?

**MP5 (numerical bug).** The -0.05pp (spirals) and -39pp (moons) deltas are artifacts of NaN training, not a genuine refutation of distance-matrix distillation. The method was never actually tested.

Additionally, the PH Wasserstein metric shows NaN for all dmat epochs after epoch 0 (because the student activations become NaN), so the claim "PH W0 decreasing fraction for dmat: 0.00" is also meaningless — it's measuring NaN, not topology.

## G5: Is the refutation from topology being useless, or implementation issues?

**Implementation issues.** Two distinct problems:

1. **Primary (blocking): sqrt gradient singularity.** The dmat loss produces NaN gradients due to zero pairwise distances. This alone invalidates the entire dmat comparison.

2. **Secondary (design): broken baseline.** Output-only distillation (MSE on raw logits) performs *worse* than standalone training on ground-truth labels in both datasets:
   - Spirals: standalone 0.60-0.78, distillation 0.49-0.59
   - Moons: standalone 0.92-1.00, distillation 0.86-0.92

   This means the teacher's raw logits (potentially large magnitude after 3000 BCE epochs) are a poor soft target when matched via MSE. Standard distillation uses temperature-scaled softmax + KL divergence (Hinton 2015), not raw logit MSE. The baseline itself is suboptimal, making any Δ measurement unreliable.

## Verdict

The REFUTED verdict is **not supported by the evidence**. The experiment tested a numerically broken loss function against a weak baseline. The hypothesis (topology-preserving distillation helps) was never actually evaluated.

## Recommendations

1. Fix `distance_matrix_loss`: use `d2.clamp(min=1e-12).sqrt()` or drop sqrt entirely.
2. Fix distillation baseline: use temperature-scaled KL divergence instead of raw logit MSE.
3. Re-run with fixes. If dmat still shows no benefit, *then* it's a genuine refutation.
4. Move item 42 back to ACTIVE with status "invalidated by MP5, needs re-run."
