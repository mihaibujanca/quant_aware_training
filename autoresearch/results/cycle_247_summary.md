# Cycle 247 — Item 257: Heterogeneous K 6-block

**Verdict: INCONCLUSIVE**

- Criteria: Confirmed ≤61.0, Refuted ≥65.0, else Inconclusive
- Config: BLOCKS=[1, 2, 4, 5, 7, 8], K_PER_BLOCK={1: 128, 2: 128, 4: 512, 5: 512, 7: 128, 8: 128}
- LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Trainable params: 1,184,256
- Float PPL: 53.895

## Per-seed results

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 103.931 | 64.621 | 2300 | 65.928 |
| 137 | 103.931 | 64.405 | 2200 | 65.373 |
| 2024 | 103.931 | 64.332 | 2400 | 65.302 |
| **mean** | 103.931 | **64.453** | 2300 | 65.534 |
| **std**  | 0.000 | 0.123 | 82 | 0.280 |

## Per-block residual MSE (lstsq init, seed 42)

| block | K | residual MSE |
|---|---|---|
| 1 | 128 | 1.2550e-01 |
| 2 | 128 | 1.7167e-01 |
| 4 | 512 | 1.6163e-01 |
| 5 | 512 | 2.2363e-01 |
| 7 | 128 | 5.8121e-01 |
| 8 | 128 | 8.1828e-01 |

## Trajectory description

### Seed 42
- init=103.93, step1100=66.52, step2100=64.89, best=step2300@64.62, step3100=64.93, step4000=65.93
### Seed 137
- init=103.93, step1100=66.73, step2100=64.94, best=step2200@64.41, step3100=65.50, step4000=65.37
### Seed 2024
- init=103.93, step1100=66.47, step2100=65.19, best=step2400@64.33, step3100=64.92, step4000=65.30

## Comparison

- Cycle 221 K=512 6-block ceiling: **59.515**
- Cycle 231 K=128 6-block floor: **67.562**
- Cycle 247 hetero K: **64.453 ± 0.123**
- Delta vs K=512 ceiling: **+4.938** PPL
- Delta vs K=128 floor:   **-3.109** PPL
- Gap recovery fraction: **0.386** (1.0 = reaches K=512 ceiling, 0.0 = matches K=128 floor)
- Param cost: 1.18M (vs K=512 6-block 2.36M, ~50%)
- Overfitting check: final@4000 mean = 65.534
