# Cycle 268 — Item 278: 4-block K=64 sweep

**Verdict: CONFIRMED**

- Criteria: CONFIRMED if min best_mean <= 56.0; REFUTED if all best_mean >= 65.0; else INCONCLUSIVE
- Subsets: [[1, 2, 4, 7], [1, 4, 7, 8], [1, 2, 7, 8]]
- Config: K=64, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895
- Wall time: 20.9 min; fits=9

## Per-subset aggregates

| subset | n_seeds | init_mean | best_mean ± std | best_step | final@4000 |
|---|---|---|---|---|---|
| [1, 2, 4, 7] | 3 | 78.258 | **56.067 ± 0.044** | 3900 ± 82 | 56.139 |
| [1, 4, 7, 8] | 3 | 78.873 | **55.953 ± 0.092** | 3967 ± 47 | 55.966 |
| [1, 2, 7, 8] | 3 | 79.628 | **57.277 ± 0.120** | 3900 ± 82 | 57.336 |

**Min over subsets: 55.953 (1_4_7_8)**

## Per-seed details

### Subset [1, 2, 4, 7]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 78.258 | 56.125 | 3900 | 56.261 |
| 137 | 78.258 | 56.060 | 3800 | 56.139 |
| 2024 | 78.258 | 56.017 | 4000 | 56.017 |

### Subset [1, 4, 7, 8]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 78.873 | 56.017 | 3900 | 56.055 |
| 137 | 78.873 | 56.020 | 4000 | 56.020 |
| 2024 | 78.873 | 55.823 | 4000 | 55.823 |

### Subset [1, 2, 7, 8]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 79.628 | 57.350 | 3900 | 57.498 |
| 137 | 79.628 | 57.372 | 3800 | 57.403 |
| 2024 | 79.628 | 57.108 | 4000 | 57.108 |

## Comparison

- Cycle 257 K=64 5-block: **62.181**
- Cycle 265 K=128 4-block min [1,2,4,7]: **53.647**
- Cycle 255 K=512 4-block min [1,2,4,7]: **49.609**
- Cycle 230 K=64 3-block: **50.781**
- Float ref: 53.895
- Min 4-block K=64 best_mean: **55.953**
- Delta vs cycle 257 K=64 5-block:  **-6.228** PPL
- Delta vs cycle 265 K=128 4-block: **+2.306** PPL
- Delta vs cycle 255 K=512 4-block: **+6.344** PPL
- Delta vs cycle 230 K=64 3-block:  **+5.172** PPL
- Delta vs float:                   **+2.058** PPL
