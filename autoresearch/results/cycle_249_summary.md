# Cycle 249 — Item 259: Heterogeneous K complement (K=512 on {7,8})

**Verdict: REFUTED**

- Criteria: Confirmed ≤61.0, Refuted ≥65.0, else Inconclusive
- Config: BLOCKS=[1, 2, 4, 5, 7, 8], K_PER_BLOCK={1: 128, 2: 128, 4: 128, 5: 128, 7: 512, 8: 512}
- LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Trainable params: 1,184,256
- Float PPL: 53.895

## Per-seed results

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 104.198 | 65.402 | 2200 | 68.893 |
| 137 | 104.198 | 65.178 | 2200 | 68.674 |
| 2024 | 104.198 | 65.244 | 2200 | 68.360 |
| **mean** | 104.198 | **65.275** | 2200 | 68.642 |
| **std**  | 0.000 | 0.094 | 0 | 0.219 |

## Per-block residual MSE (lstsq init, seed 42)

| block | K | residual MSE |
|---|---|---|
| 1 | 128 | 1.2550e-01 |
| 2 | 128 | 1.7167e-01 |
| 4 | 128 | 2.2747e-01 |
| 5 | 128 | 3.1172e-01 |
| 7 | 512 | 4.1485e-01 |
| 8 | 512 | 5.7914e-01 |

## Trajectory description

### Seed 42
- init=104.20, step1100=66.66, step2100=65.57, best=step2200@65.40, step3100=66.59, step4000=68.89
### Seed 137
- init=104.20, step1100=66.83, step2100=65.38, best=step2200@65.18, step3100=67.16, step4000=68.67
### Seed 2024
- init=104.20, step1100=66.63, step2100=65.95, best=step2200@65.24, step3100=66.96, step4000=68.36

## Comparison

- Cycle 221 K=512 6-block ceiling: **59.515**
- Cycle 231 K=128 6-block floor: **67.562**
- Cycle 247 hetero(4,5)=512: **64.453**
- Cycle 249 hetero(7,8)=512: **65.275 ± 0.094**
- Delta vs K=512 ceiling: **+5.760** PPL
- Delta vs K=128 floor:   **-2.287** PPL
- Delta vs cycle 247:     **+0.822** PPL
- Gap recovery fraction: **0.284** (1.0 = reaches K=512 ceiling, 0.0 = matches K=128 floor)
- Param cost: 1.18M (vs K=512 6-block 2.36M, ~50%)
- Overfitting check: final@4000 mean = 68.642
