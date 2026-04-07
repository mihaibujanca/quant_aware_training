# Cycle 251 — Item 261: 5-block [1,2,4,7,8] at K=128

**Verdict: INCONCLUSIVE**

- Criteria: Confirmed ≤56.0, Refuted ≥62.0, else Inconclusive
- Config: BLOCKS=[1, 2, 4, 7, 8], K=128, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895

## Per-seed results

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 88.341 | 59.111 | 3900 | 59.146 |
| 137 | 88.341 | 59.133 | 4000 | 59.133 |
| 2024 | 88.341 | 59.114 | 4000 | 59.114 |
| **mean** | 88.341 | **59.119** | 3967 | 59.131 |
| **std**  | 0.000 | 0.009 | 47 | 0.013 |

## Trajectory description

### Seed 42
- init=88.34, step1100=62.52, step2100=60.20, best=step3900@59.11, step3100=59.44, step4000=59.15
### Seed 137
- init=88.34, step1100=62.91, step2100=60.53, best=step4000@59.13, step3100=59.59, step4000=59.13
### Seed 2024
- init=88.34, step1100=62.85, step2100=60.41, best=step4000@59.11, step3100=59.80, step4000=59.11

## Comparison

- Cycle 251 5-block [1,2,4,7,8] K=128 best_mean: **59.119 ± 0.009**
- Cycle 245 5-block [1,2,4,7,8] K=512: **53.826**
- Delta vs cycle 245 K=512: **+5.293** PPL
- Cycle 231 6-block [1,2,4,5,7,8] K=128: 67.562
- Delta vs cycle 231 K=128: **-8.443** PPL
- Cycle 223 3-block [1,4,7] K=128: 48.989
- Delta vs cycle 223 K=128: **+10.130** PPL
- Float PPL baseline: 53.895
- Delta vs float: **+5.224** PPL
- Overfitting check: final@4000 mean = 59.131
