# Cycle 265 — Item 275: 4-block K=128 sweep

**Verdict: INCONCLUSIVE**

- Criteria: CONFIRMED if min best_mean ≤ 53.0; REFUTED if all best_mean ≥ 56.0; else INCONCLUSIVE
- Subsets: [[1, 2, 4, 7], [1, 4, 7, 8], [1, 2, 7, 8]]
- Config: K=128, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895
- Wall time: 21.0 min; fits=9

## Per-subset aggregates

| subset | n_seeds | init_mean | best_mean ± std | best_step | final@4000 |
|---|---|---|---|---|---|
| [1, 2, 4, 7] | 3 | 74.216 | **53.647 ± 0.066** | 3933 ± 94 | 53.706 |
| [1, 4, 7, 8] | 3 | 76.107 | **53.732 ± 0.098** | 3900 ± 82 | 53.771 |
| [1, 2, 7, 8] | 3 | 75.862 | **54.488 ± 0.137** | 3900 ± 82 | 54.561 |

**Min over subsets: 53.647 (1_2_4_7)**

## Per-seed details

### Subset [1, 2, 4, 7]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 74.216 | 53.695 | 4000 | 53.695 |
| 137 | 74.216 | 53.693 | 3800 | 53.870 |
| 2024 | 74.216 | 53.553 | 4000 | 53.553 |

### Subset [1, 4, 7, 8]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 76.107 | 53.771 | 3900 | 53.845 |
| 137 | 76.107 | 53.827 | 3800 | 53.870 |
| 2024 | 76.107 | 53.598 | 4000 | 53.598 |

### Subset [1, 2, 7, 8]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 75.862 | 54.594 | 3900 | 54.703 |
| 137 | 75.862 | 54.575 | 3800 | 54.685 |
| 2024 | 75.862 | 54.294 | 4000 | 54.294 |

## Comparison

- Cycle 255 K=512 4-block min [1,2,4,7]: **49.609**
- Cycle 255 K=512 4-block other subsets: [1,2,7,8]=50.208, [1,4,7,8]=50.712
- Cycle 251 K=128 5-block: **59.119**
- Cycle 223 K=128 3-block [1,4,7]: 48.989
- Float ref: 53.895
- Min 4-block K=128 best_mean: **53.647**
- Delta vs cycle 255 K=512 min:    **+4.038** PPL
- Delta vs cycle 251 K=128 5-block: **-5.472** PPL
- Delta vs cycle 223 K=128 3-block: **+4.658** PPL
- Delta vs float:                  **-0.248** PPL
