# Cycle 231 — Item 241: K=128 Adam 6-block correction

**Verdict: REFUTED**

- Criteria: Confirmed ≤61.0, Refuted ≥65.0, else Inconclusive
- Config: BLOCKS=[1, 2, 4, 5, 7, 8], K=128, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895

## Per-seed results

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 115.056 | 67.630 | 3900 | 67.921 |
| 137 | 115.056 | 67.483 | 3600 | 67.556 |
| 2024 | 115.056 | 67.574 | 4000 | 67.574 |
| **mean** | 115.056 | **67.562** | 3833 | 67.684 |
| **std**  | 0.000 | 0.061 | 170 | 0.168 |

## Trajectory description

### Seed 42
- init=115.06, step1100=71.49, step2100=68.85, best=step3900@67.63, step3100=67.86, step4000=67.92
### Seed 137
- init=115.06, step1100=71.75, step2100=68.79, best=step3600@67.48, step3100=68.29, step4000=67.56
### Seed 2024
- init=115.06, step1100=71.40, step2100=69.01, best=step4000@67.57, step3100=67.93, step4000=67.57

## Comparison

- Cycle 221 K=512 6-block ceiling: **59.515**
- Cycle 231 K=128 6-block best_mean: **67.562 ± 0.061**
- Delta vs K=512 ceiling: **+8.047** PPL
- Cycle 223 K=128 3-block baseline: 48.989
- Overfitting check: final@4000 mean = 67.684
