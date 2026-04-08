# Cycle 320 — ReLU² spirals/moons eff_rank census + LASER (item 329)

**Verdict:** INCONCLUSIVE

## Config
- MLP(2, 32, 1, depth=12, **ReLU²**); Adam lr=0.0001; full-batch BCEWithLogits
- N=2000; spirals(noise=0.5, n_turns=3, ×2.0); moons(noise=0.1)
- Train to acc≥0.95; cap 40000 epochs; seeds=[42, 137, 2024]
- LASER ranks=[1, 2, 3, 5]; retrain 500 Adam steps lr=0.0001 (B + bias only)
- Trainable params/net = 11745

## Verdict logic (item 329 verbatim)
- CONFIRMED-curvature-sparsifies iff init within ±0.5 of cycle 297 ReLU AND post mean strictly lower by ≥0.3 AND LASER r=2 retrain within 0.5%
- CONFIRMED-no-difference iff both init and post within ±0.3 of cycle 297 ReLU at every layer
- REFUTED-curvature-expands iff post is higher than ReLU's by ≥0.5 anywhere
- INCONCLUSIVE otherwise

## spirals

| layer | ReLU² init mean ± std | ReLU² post mean ± std | cycle 297 ReLU init | cycle 297 ReLU post |
|---:|---:|---:|---:|---:|
| 0 | 3.461 ± 0.354 | 3.385 ± 0.408 | 3.0 | 3.13 |
| 1 | 3.351 ± 0.146 | 2.336 ± 0.680 | 2.7 | 3.58 |
| 2 | 3.293 ± 0.131 | 2.128 ± 1.007 | 1.7 | 2.51 |
| 3 | 1.080 ± 0.084 | 1.299 ± 0.250 | 1.13 | 1.8 |
| 4 | 1.001 ± 0.001 | 1.180 ± 0.225 | 1.05 | 1.35 |
| 5 | 1.000 ± 0.000 | 1.073 ± 0.088 | 1.01 | 1.17 |
| 6 | 1.000 ± 0.000 | 1.006 ± 0.007 | 1.0 | 1.13 |
| 7 | 1.000 ± 0.000 | 1.000 ± 0.001 | 1.0 | 1.22 |
| 8 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.0 | 1.39 |
| 9 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.0 | 1.16 |
| 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.0 | 1.12 |
| 11 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.0 | 1.16 |

## moons

| layer | ReLU² init mean ± std | ReLU² post mean ± std | cycle 297 ReLU init | cycle 297 ReLU post |
|---:|---:|---:|---:|---:|
| 0 | 2.046 ± 0.349 | 1.975 ± 0.306 | — | — |
| 1 | 1.830 ± 0.221 | 1.280 ± 0.032 | — | — |
| 2 | 1.893 ± 0.165 | 1.064 ± 0.013 | — | — |
| 3 | 1.018 ± 0.003 | 1.036 ± 0.020 | — | — |
| 4 | 1.000 ± 0.000 | 1.063 ± 0.056 | — | — |
| 5 | 1.000 ± 0.000 | 1.078 ± 0.020 | — | — |
| 6 | 1.000 ± 0.000 | 1.012 ± 0.006 | — | — |
| 7 | 1.000 ± 0.000 | 1.000 ± 0.000 | — | — |
| 8 | 1.000 ± 0.000 | 1.000 ± 0.000 | — | — |
| 9 | 1.000 ± 0.000 | 1.000 ± 0.000 | — | — |
| 10 | 1.000 ± 0.000 | 1.000 ± 0.000 | — | — |
| 11 | 1.000 ± 0.000 | 1.000 ± 0.000 | — | — |

## LASER rank sweep (per dataset, mean across seeds)

### spirals (float held-out acc mean = 0.5217)

| rank | l_min (per seed) | no-retrain held mean | retrain best held mean |
|---:|---|---:|---:|
| 1 | [10, 10, 11] | 0.5140 | 0.5140 |
| 2 | [10, 10, 11] | 0.5140 | 0.5140 |
| 3 | [10, 10, 11] | 0.5137 | 0.5137 |
| 5 | [10, 10, 11] | 0.5160 | 0.5160 |

### moons (float held-out acc mean = 0.6070)

| rank | l_min (per seed) | no-retrain held mean | retrain best held mean |
|---:|---|---:|---:|
| 1 | [10, 9, 10] | 0.6073 | 0.6120 |
| 2 | [10, 9, 10] | 0.6070 | 0.6117 |
| 3 | [10, 9, 10] | 0.6070 | 0.6117 |
| 5 | [10, 9, 10] | 0.6073 | 0.6120 |

**Wall:** 70.7 s
