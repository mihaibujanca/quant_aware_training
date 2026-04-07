# Cycle 255 — Item 265: 4-block K=512 sweep

**Verdict: CONFIRMED**

- Criteria: CONFIRMED if min best_mean ≤ 51.0; REFUTED if all best_mean ≥ 53.0; else INCONCLUSIVE
- Subsets: [[1, 2, 4, 7], [1, 4, 7, 8], [1, 2, 7, 8]]
- Config: K=512, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895
- Wall time: 22.8 min; fits=9

## Per-subset aggregates

| subset | n_seeds | init_mean | best_mean ± std | best_step | final@4000 |
|---|---|---|---|---|---|
| [1, 2, 4, 7] | 3 | 66.826 | **49.609 ± 0.080** | 2367 ± 205 | 51.259 |
| [1, 4, 7, 8] | 3 | 68.634 | **50.712 ± 0.053** | 1733 ± 94 | 54.409 |
| [1, 2, 7, 8] | 3 | 67.896 | **50.208 ± 0.011** | 1767 ± 47 | 54.332 |

**Min over subsets: 49.609 (1_2_4_7)**

## Per-seed details

### Subset [1, 2, 4, 7]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 66.826 | 49.497 | 2100 | 51.440 |
| 137 | 66.826 | 49.678 | 2600 | 51.198 |
| 2024 | 66.826 | 49.652 | 2400 | 51.139 |

### Subset [1, 4, 7, 8]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 68.634 | 50.770 | 1800 | 54.631 |
| 137 | 68.634 | 50.642 | 1800 | 54.582 |
| 2024 | 68.634 | 50.725 | 1600 | 54.014 |

### Subset [1, 2, 7, 8]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 67.896 | 50.194 | 1700 | 54.592 |
| 137 | 67.896 | 50.214 | 1800 | 54.594 |
| 2024 | 67.896 | 50.218 | 1800 | 53.810 |

## Comparison

- Cycle 245 5-block [1,2,4,7,8] K=512: **53.826**
- Cycle 210 3-block [1,4,7] K=512: 46.556
- Float ref: 53.895
- Min 4-block best_mean: **49.609**
- Delta vs cycle 245: **-4.217** PPL
- Delta vs cycle 210: **+3.053** PPL
- Delta vs float: **-4.286** PPL
