# Cycle 250 — Item 260: 6-block [1,2,3,4,7,8] (drop 5, add 3)

**Verdict: REFUTED**

- Criteria: Confirmed ≤55.0, Refuted ≥58.0, else Inconclusive
- Config: BLOCKS=[1, 2, 3, 4, 7, 8], K=512, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895

## Per-seed results

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 85.582 | 60.049 | 1500 | 71.360 |
| 137 | 85.582 | 60.020 | 1600 | 70.850 |
| 2024 | 85.582 | 60.001 | 1600 | 70.685 |
| **mean** | 85.582 | **60.023** | 1567 | 70.965 |
| **std**  | 0.000 | 0.020 | 47 | 0.288 |

## Comparison

- Cycle 221 6-block [1,2,4,5,7,8]: **59.515**
- Cycle 245 5-block [1,2,4,7,8]:   **53.826**
- Cycle 250 6-block [1,2,3,4,7,8]: **60.023 ± 0.020**
- Delta vs c221: **+0.508** PPL
- Delta vs c245: **+6.197** PPL
- Delta vs float: **+6.128** PPL
