# Cycle 274 — Item 284: K=64 5-block extended budget (8000 steps)

**Verdict: INCONCLUSIVE**

- Criteria: CONFIRMED ∈ [61.5, 62.5] (plateau), REFUTED ≤ 60.7 (still descending), else INCONCLUSIVE
- Config: BLOCKS=[1, 2, 4, 7, 8], K=64, LR=0.0001, N_STEPS=8000, EVAL_EVERY=200, SEEDS=[42, 137, 2024]
- Float PPL: 53.895

## Per-seed results

| seed | init | best | best_step | final@8000 |
|---|---|---|---|---|
| 42 | 95.620 | 61.423 | 6200 | 61.652 |
| 137 | 95.620 | 61.314 | 6200 | 61.373 |
| 2024 | 95.620 | 61.394 | 7400 | 61.626 |
| **mean** | 95.620 | **61.377** | 6600 | 61.550 |
| **std**  | 0.000 | 0.046 | 566 | 0.126 |

## Trajectory description

### Seed 42
- init=95.62, step2200=64.17, step4200=62.17, best=step6200@61.42, step6200=61.42, step8000=61.65
### Seed 137
- init=95.62, step2200=64.09, step4200=62.05, best=step6200@61.31, step6200=61.31, step8000=61.37
### Seed 2024
- init=95.62, step2200=64.21, step4200=62.26, best=step7400@61.39, step6200=61.48, step8000=61.63

## Comparison

- Cycle 274 K=64 5-block 8k best_mean: **61.377 ± 0.046**
- Cycle 257 K=64 5-block 4k baseline: 62.181
- Delta vs cycle 257 4k: **-0.804** PPL
- Cycle 254 K=128 5-block 8k (plateau precedent): 59.018
- Delta vs cycle 254: **+2.359** PPL
- Cycle 251 K=128 5-block 4k: 59.119
- Cycle 245 K=512 5-block: 53.826
- Float PPL baseline: 53.895
- Delta vs float: **+7.482** PPL
- Overfitting check: final@8000 mean = 61.550 (Δ vs best = +0.173)
