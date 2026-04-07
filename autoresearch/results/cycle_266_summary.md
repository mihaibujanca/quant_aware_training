# Cycle 266 — Item 276: 2-block K=512 sweep

**Verdict: CONFIRMED**

- Criteria: CONFIRMED if min best_mean ≤ 45.0; REFUTED if all best_mean ≥ 47.0; else INCONCLUSIVE
- Subsets: [[1, 4], [1, 7], [4, 7]]
- Config: K=512, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895
- Wall time: 22.2 min; fits=9

## Per-subset aggregates

| subset | n_seeds | init_mean | best_mean ± std | best_step | final@4000 |
|---|---|---|---|---|---|
| [1, 4] | 3 | 57.057 | **43.852 ± 0.027** | 3900 ± 82 | 43.864 |
| [1, 7] | 3 | 58.018 | **43.905 ± 0.051** | 3933 ± 94 | 43.907 |
| [4, 7] | 3 | 59.777 | **45.770 ± 0.016** | 3867 ± 94 | 45.807 |

**Min over subsets: 43.852 (1_4)**

## Per-seed details

### Subset [1, 4]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 57.057 | 43.866 | 3900 | 43.898 |
| 137 | 57.057 | 43.877 | 4000 | 43.877 |
| 2024 | 57.057 | 43.814 | 3800 | 43.819 |

### Subset [1, 7]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 58.018 | 43.946 | 4000 | 43.946 |
| 137 | 58.018 | 43.833 | 4000 | 43.833 |
| 2024 | 58.018 | 43.935 | 3800 | 43.943 |

### Subset [4, 7]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 59.777 | 45.752 | 4000 | 45.752 |
| 137 | 59.777 | 45.767 | 3800 | 45.775 |
| 2024 | 59.777 | 45.792 | 3800 | 45.893 |

## Comparison

- Cycle 210 K=512 3-block [1,4,7]: **46.556**
- Cycle 255 K=512 4-block min:    49.609
- Cycle 245 K=512 5-block:        53.826
- Cycle 221 K=512 6-block:        59.515
- Float ref:                       53.895
- Min 2-block best_mean:          **43.852**
- Delta vs cycle 210 (3b):        **-2.704** PPL
- Delta vs cycle 255 (4b min):    **-5.757** PPL
- Delta vs float:                 **-10.042** PPL
