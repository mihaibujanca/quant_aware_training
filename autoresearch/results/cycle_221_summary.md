# Cycle 221 — Item 226: Adam 6-block + early stopping

**Verdict: INCONCLUSIVE**

- Criteria: Confirmed ≤58.0, Refuted >62.0, else Inconclusive
- Config: BLOCKS=[1, 2, 4, 5, 7, 8], K=512, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895

## Per-seed results

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 86.686 | 59.391 | 1300 | 72.250 |
| 137 | 86.686 | 59.620 | 1200 | 70.553 |
| 2024 | 86.686 | 59.533 | 1600 | 70.359 |
| **mean** | 86.686 | **59.515** | 1367 | 71.054 |
| **std**  | 0.000 | 0.094 | 170 | 0.850 |

## Trajectory description

### Seed 42
- init=86.69, step1100=59.75, step2100=60.83, best=step1300@59.39, step3100=65.57, step4000=72.25
### Seed 137
- init=86.69, step1100=59.98, step2100=60.53, best=step1200@59.62, step3100=65.93, step4000=70.55
### Seed 2024
- init=86.69, step1100=60.06, step2100=61.08, best=step1600@59.53, step3100=65.72, step4000=70.36

## Comparison

- Cycle 211 single-seed best (implicit, step 1500): **59.680**
- Cycle 221 3-seed best_mean: **59.515 ± 0.094**
- Delta: **-0.165** PPL
- 3-block [1,4,7] Adam baseline (cycle 206): 46.556
- Overfitting check: final@4000 mean = 71.054 (cycle 211 single-seed: 72.250)
