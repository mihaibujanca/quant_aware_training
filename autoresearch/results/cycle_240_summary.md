# Cycle 240 — Item 250: K=1 Adam correction at 6 blocks

**Verdict: REFUTED**

- Criteria: CONFIRMED if best_mean ∈ [58.0, 62.0]; REFUTED if best_mean ≥ 70.0; else INCONCLUSIVE.
- Config: BLOCKS=[1, 2, 4, 5, 7, 8], K=1, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Trainable params: 9,216 (6 × (768 + 768))
- Float PPL: 53.895

## Per-seed results

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 203.246 | 96.943 | 4000 | 96.943 |
| 137 | 203.246 | 96.786 | 4000 | 96.786 |
| 2024 | 203.246 | 96.929 | 4000 | 96.929 |
| **mean** | 203.246 | **96.886** | 4000 | 96.886 |
| **std**  | 0.000 | 0.071 | 0 | 0.071 |

## Trajectory description

### Seed 42
- init=203.25, step1100=110.81, step2100=102.43, best=step4000@96.94, step3100=98.63, step4000=96.94
### Seed 137
- init=203.25, step1100=110.95, step2100=102.47, best=step4000@96.79, step3100=98.71, step4000=96.79
### Seed 2024
- init=203.25, step1100=111.21, step2100=102.92, best=step4000@96.93, step3100=98.74, step4000=96.93

## Comparison

| config | best_ppl_mean | source |
|---|---|---|
| 6-block K=512 (baseline)   | 59.515 | cycle 221 |
| 6-block K=128              | 67.562 | cycle 231 |
| 6-block K=1   (this cycle) | **96.886 ± 0.071** | cycle 240 |
| 3-block K=1   (reference)  | 59.245 | cycle 230 |

- Δ vs cycle 221 K=512 6-block: **+37.371** PPL
- Δ vs cycle 231 K=128 6-block: **+29.324** PPL
- Δ vs cycle 230 K=1 3-block:   **+37.641** PPL
