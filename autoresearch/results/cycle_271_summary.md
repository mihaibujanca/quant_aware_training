# Cycle 271 — Item 281: 1-block K=512 sweep

**Verdict: INCONCLUSIVE**

- Criteria: CONFIRMED if min best_mean ≤ 43.0; REFUTED if all best_mean ≥ 46.0; else INCONCLUSIVE
- Subsets: [[1], [4], [7]]
- Config: K=512, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895
- Wall time: 21.7 min; fits=9

## Per-subset aggregates

| subset | n_seeds | init_mean | best_mean ± std | best_step | final@4000 |
|---|---|---|---|---|---|
| [1] | 3 | 54.689 | **43.820 ± 0.023** | 4000 ± 0 | 43.820 |
| [4] | 3 | 55.715 | **44.633 ± 0.027** | 3967 ± 47 | 44.636 |
| [7] | 3 | 56.774 | **45.375 ± 0.003** | 4000 ± 0 | 45.375 |

**Min over subsets: 43.820 (1)**

## Per-seed details

### Subset [1]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 54.689 | 43.826 | 4000 | 43.826 |
| 137 | 54.689 | 43.846 | 4000 | 43.846 |
| 2024 | 54.689 | 43.789 | 4000 | 43.789 |

### Subset [4]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 55.715 | 44.635 | 3900 | 44.642 |
| 137 | 55.715 | 44.665 | 4000 | 44.665 |
| 2024 | 55.715 | 44.599 | 4000 | 44.599 |

### Subset [7]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 56.774 | 45.377 | 4000 | 45.377 |
| 137 | 56.774 | 45.371 | 4000 | 45.371 |
| 2024 | 56.774 | 45.376 | 4000 | 45.376 |

## Comparison

- Cycle 266 K=512 2-block [1,4]:  **43.852**
- Cycle 210 K=512 3-block [1,4,7]: 46.556
- Cycle 255 K=512 4-block min:    49.609
- Cycle 245 K=512 5-block:        53.826
- Cycle 221 K=512 6-block:        59.515
- Float ref:                       53.895
- Min 1-block best_mean:          **43.820**
- Delta vs cycle 266 (2b [1,4]):  **-0.032** PPL
- Delta vs cycle 210 (3b):        **-2.736** PPL
- Delta vs float:                 **-10.074** PPL
