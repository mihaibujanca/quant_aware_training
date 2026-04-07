# Cycle 257 — Item 267: 5-block [1,2,4,7,8] at K=64

**Verdict: CONFIRMED**

- Criteria: Confirmed ≤64.0, Refuted ≥75.0, else Inconclusive
- Config: BLOCKS=[1, 2, 4, 7, 8], K=64, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895

## Per-seed results

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 95.620 | 62.212 | 3900 | 62.241 |
| 137 | 95.620 | 62.106 | 4000 | 62.106 |
| 2024 | 95.620 | 62.224 | 4000 | 62.224 |
| **mean** | 95.620 | **62.181** | 3967 | 62.190 |
| **std**  | 0.000 | 0.053 | 47 | 0.060 |

## Trajectory description

### Seed 42
- init=95.62, step1100=67.04, step2100=64.06, best=step3900@62.21, step3100=62.80, step4000=62.24
### Seed 137
- init=95.62, step1100=67.38, step2100=64.32, best=step4000@62.11, step3100=62.83, step4000=62.11
### Seed 2024
- init=95.62, step1100=67.41, step2100=64.34, best=step4000@62.22, step3100=63.08, step4000=62.22

## Comparison

- Cycle 257 5-block [1,2,4,7,8] K=64 best_mean: **62.181 ± 0.053**
- Cycle 245 5-block [1,2,4,7,8] K=512: **53.826**
- Delta vs cycle 245 K=512: **+8.355** PPL
- Cycle 251 5-block [1,2,4,7,8] K=128: 59.119
- Delta vs cycle 251 K=128: **+3.062** PPL
- Cycle 230 3-block K=64: 50.781
- Delta vs cycle 230 K=64 3-block: **+11.400** PPL
- Float PPL baseline: 53.895
- Delta vs float: **+8.286** PPL
- Overfitting check: final@4000 mean = 62.190
