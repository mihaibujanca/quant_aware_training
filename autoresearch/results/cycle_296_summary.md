# Cycle 296 — LASER recipe on spirals/moons (item 306)

**Verdict (spirals): INCONCLUSIVE**

## Config
- MLP(2, 32, 1, depth=12, ReLU); Adam train lr=0.001; full-batch BCEWithLogits
- N=2000 (1000 train / 1000 held-out per seed)
- spirals(noise=0.5, n_turns=3, input_scale=2.0); moons(noise=0.1)
- Stop training when train_acc >= 0.95 (cap 20000 epochs)
- LASER: select l* = argmin eff_rank over l in [2..11]; SVD W (32x32) at l*; ranks [1, 2, 3, 5]
- Retrain: Adam lr=0.001, 500 full-batch steps on TRAIN, eval HELD-OUT every 25, best-state restore
- Trainable per fit: 32(r+1); seeds [42, 137, 2024]

## Verdict logic (item 306 verbatim, applied to spirals)
- CONFIRMED iff rank-2 retrain held-out acc >= float held-out acc - 0.5% on spirals AND no-retrain rank-2 >= 80% of float held-out acc
- REFUTED iff rank-2 retrain held-out gap > 5% from float on spirals
- INCONCLUSIVE between

## spirals (verdict: INCONCLUSIVE)

- float held-out acc (mean ± std): **0.8270 ± 0.0205**

| r | no_retrain_mean | no_retrain_std | retrain_mean | retrain_std | gap_vs_float |
|--:|----------------:|---------------:|-------------:|------------:|-------------:|
| 1 | 0.5307 | 0.0695 | 0.7830 | 0.0453 | +0.0440 |
| 2 | 0.6363 | 0.0856 | 0.8207 | 0.0303 | +0.0063 |
| 3 | 0.6850 | 0.0840 | 0.8023 | 0.0312 | +0.0247 |
| 5 | 0.6997 | 0.0956 | 0.8307 | 0.0245 | -0.0037 |

- rank-2 retrain gap vs float: **+0.0063**
- rank-2 no-retrain frac of float: **0.7694**

## moons (verdict: CONFIRMED)

- float held-out acc (mean ± std): **0.9967 ± 0.0019**

| r | no_retrain_mean | no_retrain_std | retrain_mean | retrain_std | gap_vs_float |
|--:|----------------:|---------------:|-------------:|------------:|-------------:|
| 1 | 0.7943 | 0.2168 | 0.9963 | 0.0017 | +0.0003 |
| 2 | 0.9947 | 0.0025 | 0.9967 | 0.0021 | +0.0000 |
| 3 | 0.9950 | 0.0022 | 0.9970 | 0.0022 | -0.0003 |
| 5 | 0.9967 | 0.0019 | 0.9973 | 0.0024 | -0.0007 |

- rank-2 retrain gap vs float: **+0.0000**
- rank-2 no-retrain frac of float: **0.9980**

## Per-seed details
- spirals seed=42: epochs=5425 train_acc=0.9510 float_held=0.8410 l*=5 eff_rank[l*]=1.423
  - r=1: no_retrain=0.6280  retrain_best=0.7840@375
  - r=2: no_retrain=0.5500  retrain_best=0.8390@350
  - r=3: no_retrain=0.5910  retrain_best=0.7750@500
  - r=5: no_retrain=0.7570  retrain_best=0.8470@25
- spirals seed=137: epochs=3725 train_acc=0.9530 float_held=0.7980 l*=6 eff_rank[l*]=1.195
  - r=1: no_retrain=0.4700  retrain_best=0.7270@475
  - r=2: no_retrain=0.6060  retrain_best=0.7780@500
  - r=3: no_retrain=0.6690  retrain_best=0.7860@225
  - r=5: no_retrain=0.7770  retrain_best=0.7960@25
- spirals seed=2024: epochs=2775 train_acc=0.9530 float_held=0.8420 l*=4 eff_rank[l*]=1.447
  - r=1: no_retrain=0.4940  retrain_best=0.8380@275
  - r=2: no_retrain=0.7530  retrain_best=0.8450@50
  - r=3: no_retrain=0.7950  retrain_best=0.8460@50
  - r=5: no_retrain=0.5650  retrain_best=0.8490@100
- moons seed=42: epochs=100 train_acc=0.9880 float_held=0.9980 l*=3 eff_rank[l*]=1.136
  - r=1: no_retrain=0.8960  retrain_best=0.9970@50
  - r=2: no_retrain=0.9920  retrain_best=0.9970@25
  - r=3: no_retrain=0.9930  retrain_best=0.9980@500
  - r=5: no_retrain=0.9980  retrain_best=0.9990@425
- moons seed=137: epochs=100 train_acc=1.0000 float_held=0.9980 l*=4 eff_rank[l*]=1.094
  - r=1: no_retrain=0.4930  retrain_best=0.9980@125
  - r=2: no_retrain=0.9980  retrain_best=0.9990@25
  - r=3: no_retrain=0.9980  retrain_best=0.9990@25
  - r=5: no_retrain=0.9980  retrain_best=0.9990@25
- moons seed=2024: epochs=100 train_acc=0.9990 float_held=0.9940 l*=11 eff_rank[l*]=1.128
  - r=1: no_retrain=0.9940  retrain_best=0.9940@0
  - r=2: no_retrain=0.9940  retrain_best=0.9940@0
  - r=3: no_retrain=0.9940  retrain_best=0.9940@0
  - r=5: no_retrain=0.9940  retrain_best=0.9940@0

Total wall: 15.9s
