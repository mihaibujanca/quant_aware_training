# Cycle 272 — Item 282: 4-block K=256 sweep

**Verdict: CONFIRMED**

- Criteria: CONFIRMED if min best_mean ≤ 52.0; REFUTED if all best_mean ≥ 54.0; else INCONCLUSIVE
- Subsets: [[1, 2, 4, 7], [1, 4, 7, 8], [1, 2, 7, 8]]
- Config: K=256, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895
- Wall time: 21.3 min; fits=9

## Per-subset aggregates

| subset | n_seeds | init_mean | best_mean ± std | best_step | final@4000 |
|---|---|---|---|---|---|
| [1, 2, 4, 7] | 3 | 70.869 | **51.922 ± 0.020** | 3633 ± 170 | 52.070 |
| [1, 4, 7, 8] | 3 | 72.818 | **52.472 ± 0.059** | 3233 ± 368 | 52.778 |
| [1, 2, 7, 8] | 3 | 72.450 | **52.676 ± 0.083** | 3100 ± 294 | 53.050 |

**Min over subsets: 51.922 (1_2_4_7)**

## Per-seed details

### Subset [1, 2, 4, 7]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 70.869 | 51.951 | 3700 | 52.063 |
| 137 | 70.869 | 51.907 | 3800 | 52.122 |
| 2024 | 70.869 | 51.909 | 3400 | 52.026 |

### Subset [1, 4, 7, 8]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 72.818 | 52.549 | 3700 | 52.840 |
| 137 | 72.818 | 52.460 | 2800 | 52.906 |
| 2024 | 72.818 | 52.406 | 3200 | 52.589 |

### Subset [1, 2, 7, 8]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 72.450 | 52.781 | 3500 | 53.125 |
| 137 | 72.450 | 52.669 | 2800 | 53.112 |
| 2024 | 72.450 | 52.579 | 3000 | 52.914 |

## Comparison

- Cycle 255 K=512 4-block min [1,2,4,7]: **49.609**
- Cycle 265 K=128 4-block min [1,2,4,7]: **53.647**
- Cycle 223 K=256 3-block [1,4,7]: 47.509
- Cycle 256 K=256 6-block: 64.181
- Float ref: 53.895
- Min 4-block K=256 best_mean: **51.922**
- Delta vs cycle 255 K=512 4-block: **+2.313** PPL
- Delta vs cycle 265 K=128 4-block: **-1.725** PPL
- Delta vs cycle 223 K=256 3-block: **+4.413** PPL
- Delta vs float:                  **-1.972** PPL
