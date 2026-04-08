# Cycle 298 — SDF rank census + LASER replication (item 308)

**Verdict: REFUTED**

## Config
- MLP(3, 256, 1, depth=4, ReLU); Adam lr=0.001; full-batch MSE
- mesh=torus; N=30000 points (train_frac=0.8)
- surface_ratio=0.5, surface_noise=0.01, seed=42
- max_epochs=5000, plateau_patience=250
- LASER: l* = argmin eff_rank over l in [1..3]; SVD W (256x256); ranks [2, 3, 5, 10]
- Retrain: 500 Adam steps lr=0.001, eval held every 25
- Trainable per fit: 256(r+1)

## Verdict logic (item 308 verbatim)
- CONFIRMED iff (a) hidden eff_rank ∈ [2.5, 3.5] AND (b) grad eff_rank ∈ [1.8, 2.5] AND (c) rank-3 retrain MSE ≤ 1.1× float MSE
- REFUTED iff hidden eff_rank ≥ 8 OR rank-3 retrain MSE ≥ 2× float
- INCONCLUSIVE between

## Training
- final train MSE: 2.370525e-06
- float held-out MSE: **3.309283e-06**
- best epoch: 2925 (plateau-stop or max)
- params: 198657

## Hidden eff_rank census
- layer 0: eff_rank = 3.9022
- layer 1: eff_rank = 5.0660
- layer 2: eff_rank = 3.0428
- layer 3: eff_rank = 1.8617 ← l*
- mean over hidden 1..3: **3.3235** (target [2.5, 3.5])

## Gradient ensemble (jacobian_rank_profile)
- effective_rank: **2.6226** (target [1.8, 2.5])
- rank_90: 3, rank_95: 3, rank_99: 3
- top singular values: [117.3413, 70.0661, 69.1223]
- n_samples: 24000, input_dim: 3

## LASER sweep at l* = 3
- SVD top-10 of W_l*: [3.2942, 2.3829, 1.5206, 1.3984, 1.2225, 1.1788, 1.1614, 1.1435, 1.1352, 1.1115]

| r | no_retrain_held_mse | retrain_best_held_mse | retrain_best_step | ratio_vs_float |
|--:|--------------------:|----------------------:|------------------:|---------------:|
| 2 | 6.121082e-03 | 6.839405e-06 | 500 | 2.0667 |
| 3 | 7.583317e-03 | 6.679289e-06 | 500 | 2.0183 |
| 5 | 2.789493e-03 | 5.574008e-06 | 500 | 1.6844 |
| 10 | 1.569573e-03 | 4.317571e-06 | 500 | 1.3047 |

## Verdict reasoning
- hidden mean eff_rank = 3.3235 in [2.5, 3.5]? **True**
- grad eff_rank = 2.6226 in [1.8, 2.5]? **False**
- rank-3 retrain ratio = 2.0183 ≤ 1.1? **False**

**Final verdict: REFUTED**

Total wall: 71.6s
