# Task
Topology-preserving distillation on toy ReLU networks (item 42). Distill teacher into a structurally different student using distance-matrix-matching as a differentiable topology proxy.

# Hypothesis
- **Confirmed if**: distance-matrix-distilled student >= 3pp higher test accuracy than output-only AND distance-matrix error decreases monotonically.
- **Refuted if**: distance-matrix loss doesn't converge OR topo-distilled <= output-only across all lambdas.
- **Inconclusive if**: distance-matrix converges but accuracy gain < 2pp.

# Pre-flight
- P1 PASS: no distillation experiment in PAST EXPERIMENTS.
- P2 PASS: no dependencies.
- P5 revised: output-only + CKA-matching + activation-MSE baselines.
- P6 revised: distance-matrix matching (Frobenius) replaces PH-weighted MSE — fully differentiable, no torch-tda. PH via ripser for post-hoc eval only.
- P7: 3 seeds, 2 datasets (moons + spirals).

# Steps
1. File: `experiments/topo_preserving_distillation.py`.
2. Teacher: 2→32×4→1 ReLU, train on spirals (n=2000, 3 turns, scale=2.0, lr=0.001) and moons. Verify test acc >= 92%.
3. Student: **2→32→1** ReLU (1 hidden layer, matched width, structurally different from 4-layer teacher).
4. **Layer alignment**: student has 1 hidden layer, teacher has 4. Align student hidden → teacher layer 2 (middle). Project both to 2D via PCA fitted on teacher activations before computing distances.
5. **Distance-matrix loss (differentiable)**: For minibatch of 200 points, compute pairwise Euclidean distance matrices D_t, D_s from PCA-projected activations. L_dmat = ||D_t - D_s||_F / (n*(n-1)). Fully differentiable through student path.
6. **Distillation variants** (4 methods):
   - (A) Output-only: L = MSE(y_s, y_t)
   - (B) Distance-matrix: L = MSE(y_s, y_t) + λ * L_dmat
   - (C) CKA-matching: L = MSE(y_s, y_t) + λ * (1 - linear_CKA(H_s, H_t))
   - (D) Activation-MSE: L = MSE(y_s, y_t) + λ * MSE(PCA(h_s), PCA(h_t))
   Sweep λ ∈ {0.1, 0.5, 1.0, 2.0} for B/C/D.
7. Train student 2000 epochs, Adam lr=0.001. 3 seeds [42, 123, 7]. Both datasets.
8. **Post-hoc PH** (not in training loop): at epochs {0, 500, 1000, 1500, 2000}, compute PD via ripser on PCA-projected 2D activations, report Wasserstein-1 for H0/H1.
9. **Metrics**: (a) test accuracy per method/λ, (b) L_dmat convergence curve, (c) post-hoc PD Wasserstein, (d) accuracy delta vs output-only.
10. Validation: teacher acc >= 92%, output-only student acc >= 70%.
11. Save: `autoresearch/results/cycle_064_topo_distillation.json`, 2-panel figure (accuracy bars across methods; PD Wasserstein through training).

# Fallbacks
- If PCA to 2D loses >40% variance: use 3D, or skip PCA (32D→32D distance matrices, same dim).
- If distance-matrix OOMs at 200 points: subsample to 100.
- If all topo methods ≈ output-only: report negative result.
