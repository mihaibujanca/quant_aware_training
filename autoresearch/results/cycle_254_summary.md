# Cycle 254 — Item 264: 5-block [1,2,4,7,8] at K=128, 8000 steps

**Verdict: REFUTED**

- Criteria: Confirmed ≤56.0, Refuted ≥58.5, else Inconclusive
- Config: BLOCKS=[1, 2, 4, 7, 8], K=128, LR=0.0001, N_STEPS=8000, EVAL_EVERY=200, SEEDS=[42, 137, 2024]
- Float PPL: 53.895

## Per-seed results

| seed | init | best | best_step | final@8000 |
|---|---|---|---|---|
| 42 | 88.341 | 58.996 | 4800 | 60.614 |
| 137 | 88.341 | 58.982 | 4400 | 60.416 |
| 2024 | 88.341 | 59.076 | 5200 | 60.520 |
| **mean** | 88.341 | **59.018** | 4800 | 60.517 |
| **std**  | 0.000 | 0.041 | 327 | 0.081 |

## Trajectory description

### Seed 42
- init=88.34, step2200=60.30, step4200=59.09, best=step4800@59.00, step6200=59.37, step8000=60.61
### Seed 137
- init=88.34, step2200=60.33, step4200=59.11, best=step4400@58.98, step6200=59.29, step8000=60.42
### Seed 2024
- init=88.34, step2200=60.41, step4200=59.32, best=step5200@59.08, step6200=59.43, step8000=60.52

## Comparison

- Cycle 254 5-block [1,2,4,7,8] K=128 8k best_mean: **59.018 ± 0.041**
- Cycle 251 5-block [1,2,4,7,8] K=128 4k: **59.119**
- Delta vs cycle 251 (4k → 8k): **-0.101** PPL
- Cycle 245 5-block [1,2,4,7,8] K=512: **53.826**
- Delta vs cycle 245 K=512: **+5.192** PPL
- Cycle 231 6-block [1,2,4,5,7,8] K=128: 67.562
- Delta vs cycle 231 K=128: **-8.544** PPL
- Cycle 223 3-block [1,4,7] K=128: 48.989
- Delta vs cycle 223 K=128: **+10.029** PPL
- Float PPL baseline: 53.895
- Delta vs float: **+5.123** PPL
- Overfitting check: final@8000 mean = 60.517
