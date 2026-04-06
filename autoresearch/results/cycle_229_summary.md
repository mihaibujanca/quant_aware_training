# Cycle 229 — Item 239: Sequential 6-block Adam W_out correction

**Verdict: REFUTED**

- Criteria: Confirmed `final_mean <= 58.0`; Refuted `final_mean >= 60.0`; Inconclusive otherwise.
- Config: BLOCKS=[1, 2, 4, 5, 7, 8], K=512, LR=0.0001, N_STEPS_PER_BLOCK=2000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895

## Per-seed table

| seed | init | after_b1 | after_b2 | after_b4 | after_b5 | after_b7 | after_b8 | final |
|---|---|---|---|---|---|---|---|---|
| 42 | 86.686 | 66.354 | 62.680 | 61.286 | 60.678 | 60.034 | 59.949 | 59.949 |
| 137 | 86.686 | 66.254 | 62.882 | 61.523 | 60.810 | 60.166 | 60.144 | 60.144 |
| 2024 | 86.686 | 66.336 | 62.910 | 61.364 | 60.612 | 60.179 | 60.119 | 60.119 |
| **mean** | 86.686 | 66.314 | 62.824 | 61.391 | 60.700 | 60.126 | 60.071 | **60.071** |
| **std**  | 0.000 | 0.044 | 0.102 | 0.099 | 0.083 | 0.065 | 0.086 | 0.086 |

## Trajectory description

### Seed 42
- init=86.69, b1=66.35@2000, b2=62.68@1900, b4=61.29@1900, b5=60.68@1600, b7=60.03@2000, b8=59.95@800, final=59.95
### Seed 137
- init=86.69, b1=66.25@2000, b2=62.88@2000, b4=61.52@1900, b5=60.81@1800, b7=60.17@1700, b8=60.14@700, final=60.14
### Seed 2024
- init=86.69, b1=66.34@2000, b2=62.91@2000, b4=61.36@2000, b5=60.61@1800, b7=60.18@1800, b8=60.12@400, final=60.12

## Comparison

- Sequential 6-block (this cycle): **60.071 ± 0.086**
- Joint 6-block (cycle 221, best_mean): **59.515**  → Δ = **+0.556** PPL
- Sequential 3-block (cycle 215, final_mean): **47.253**  → Δ = **+12.818** PPL
- Joint 3-block (cycle 210, best): **46.556**  → Δ = **+13.515** PPL
- Float PPL: **53.895**
