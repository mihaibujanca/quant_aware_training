# Cycle 286 — Item 296: 1-block [2] K=512

**Verdict: INCONCLUSIVE**

- Criteria: CONFIRMED iff best_mean ≤ 43.5; REFUTED iff best_mean ≥ 44.5; else INCONCLUSIVE
- Subset: [2], K=512
- Config: LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895
- Wall time: 7.8 min; fits=3

## Aggregate

| subset | params | n_seeds | init_mean | best_mean ± std | best_step | final@4000 |
|---|---|---|---|---|---|---|
| [2] | 393,984 | 3 | 55.461 | **43.563 ± 0.026** | 4000 ± 0 | 43.563 |

## Per-seed details

### Subset [2]

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 55.461 | 43.530 | 4000 | 43.530 |
| 137 | 55.461 | 43.593 | 4000 | 43.593 |
| 2024 | 55.461 | 43.564 | 4000 | 43.564 |

## Comparison

- Cycle 271 [1] K=512:        **43.821** (393,984 params)
- Cycle 271 [4] K=512:        44.630
- Cycle 271 [7] K=512:        45.370
- Cycle 283 [2] K=64:         47.511 (49,920 params)
- Cycle 283 [2] K=32:         47.897
- Cycle 283 [2] K=16:         49.462
- Float ref:                  53.895
- Cycle 286 [2] K=512:        **43.563** (393,984 params)
- Delta vs cycle 271 [1] K=512:  **-0.258** PPL
- Delta vs cycle 283 [2] K=64:   **-3.949** PPL
- Delta vs float:                **-10.332** PPL
