# Cycle 245 — Item 255: Drop block 5 from 6-block compression

**Verdict: CONFIRMED**

- Criteria: Confirmed ≤54.0, Refuted ≥57.0, else Inconclusive
- Config: BLOCKS=[1, 2, 4, 7, 8], K=512, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895

## Per-seed results

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 74.851 | 53.868 | 1700 | 60.454 |
| 137 | 74.851 | 53.789 | 1600 | 59.961 |
| 2024 | 74.851 | 53.823 | 1600 | 60.270 |
| **mean** | 74.851 | **53.826** | 1633 | 60.228 |
| **std**  | 0.000 | 0.032 | 47 | 0.204 |

## Trajectory description

### Seed 42
- init=74.85, step1100=54.18, step2100=54.27, best=step1700@53.87, step3100=56.63, step4000=60.45
### Seed 137
- init=74.85, step1100=54.44, step2100=54.47, best=step1600@53.79, step3100=56.85, step4000=59.96
### Seed 2024
- init=74.85, step1100=54.37, step2100=54.54, best=step1600@53.82, step3100=57.19, step4000=60.27

## Comparison

- Cycle 221 6-block [1,2,4,5,7,8] best_mean: **59.515**
- Cycle 245 5-block [1,2,4,7,8] best_mean: **53.826 ± 0.032**
- Delta vs cycle 221: **-5.689** PPL
- Cycle 210 3-block [1,4,7] baseline: 46.556
- Delta vs cycle 210: **+7.270** PPL
- Overfitting check: final@4000 mean = 60.228
