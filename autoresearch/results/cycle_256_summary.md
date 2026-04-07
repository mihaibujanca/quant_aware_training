# Cycle 256 — Item 266: 6-block K=256

**Verdict: INCONCLUSIVE**

- Criteria (agenda item 266): Confirmed ≤63.0, Refuted ≥65.0, else Inconclusive
- Config: BLOCKS=[1, 2, 4, 5, 7, 8], K=256, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895

## Per-seed results

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 101.088 | 64.220 | 2300 | 66.577 |
| 137 | 101.088 | 64.182 | 2200 | 65.848 |
| 2024 | 101.088 | 64.141 | 2400 | 65.847 |
| **mean** | 101.088 | **64.181** | 2300 | 66.091 |
| **std**  | 0.000 | 0.032 | 82 | 0.344 |

## Trajectory description

### Seed 42
- init=101.09, step1100=65.68, step2100=64.42, best=step2300@64.22, step3100=64.99, step4000=66.58
### Seed 137
- init=101.09, step1100=65.89, step2100=64.52, best=step2200@64.18, step3100=65.37, step4000=65.85
### Seed 2024
- init=101.09, step1100=65.86, step2100=64.85, best=step2400@64.14, step3100=65.09, step4000=65.85

## Comparison

- Cycle 221 K=512 6-block: **59.515**
- Cycle 231 K=128 6-block: **67.562**
- Cycle 256 K=256 6-block: **64.181 ± 0.032**
- Cycle 247 hetero K512@{4,5}: 64.453
- Cycle 249 hetero K512@{7,8}: 65.275
- Delta vs K=512 c221: **+4.666** PPL
- Delta vs K=128 c231: **-3.381** PPL
- Delta vs float: **+10.286** PPL
