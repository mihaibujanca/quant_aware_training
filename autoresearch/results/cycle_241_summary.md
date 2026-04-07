# Cycle 241 — Item 251: Adam W_out+W_fc-bias 6-block

**Verdict: REFUTED**

- Criteria: Confirmed ≤58.0, Refuted ∈[59.0,60.5], else Inconclusive
- Config: BLOCKS=[1, 2, 4, 5, 7, 8], K=512, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Trainable surface: W_out + b_out + full c_fc bias (3072,) — W_fc matrix FROZEN
- Trainable params: 2,382,336 (2.382M)
- Float PPL (held-out): 53.895

## Per-seed results

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 86.686 | 59.385 | 1300 | 73.287 |
| 137 | 86.686 | 59.611 | 1200 | 71.449 |
| 2024 | 86.686 | 59.602 | 1600 | 71.108 |
| **mean** | 86.686 | **59.533** | 1367 | 71.948 |
| **std**  | 0.000 | 0.104 | 170 | 0.957 |

## Trajectory description

### Seed 42
- init=86.69, step1100=59.73, step2100=60.94, best=step1300@59.39, step3100=65.95, step4000=73.29
### Seed 137
- init=86.69, step1100=59.98, step2100=60.66, best=step1200@59.61, step3100=66.38, step4000=71.45
### Seed 2024
- init=86.69, step1100=60.06, step2100=61.24, best=step1600@59.60, step3100=66.21, step4000=71.11

## Comparison

- Cycle 221 W_out-only 6-block baseline: **59.515 ± 0.094**
- Cycle 241 W_out+W_fc-bias 6-block best_mean: **59.533 ± 0.104**
- Delta vs cycle 221: **+0.018** PPL
- Cycle 213 W_out+W_fc-bias 3-block baseline: 46.424
- Overfit check: final@4000 mean = 71.948
