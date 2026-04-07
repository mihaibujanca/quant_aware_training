# Cycle 287 — Item 297: 1-block [1] K=512 extended budget (8000 steps)

**Verdict: CONFIRMED**

- Criteria: CONFIRMED if best_mean ≤ 43.0; REFUTED if best_mean ≥ 43.7; else INCONCLUSIVE
- Subsets: [[1]]
- Config: K=512, LR=0.0001, N_STEPS=8000, EVAL_EVERY=200, SEEDS=[42, 137, 2024]
- Float PPL: 53.895
- Wall time: 13.0 min; fits=3

## Per-subset aggregates

| subset | n_seeds | init_mean | best_mean ± std | best_step | final@8000 |
|---|---|---|---|---|---|
| [1] | 3 | 54.689 | **42.785 ± 0.013** | 8000 ± 0 | 42.785 |

**Min over subsets: 42.785 (1)**

## Per-seed details

### Subset [1]

| seed | init | best | best_step | final@8000 |
|---|---|---|---|---|
| 42 | 54.689 | 42.787 | 8000 | 42.787 |
| 137 | 54.689 | 42.800 | 8000 | 42.800 |
| 2024 | 54.689 | 42.769 | 8000 | 42.769 |

## Comparison

- Cycle 271 [1] K=512 4k:  **43.820**  (393,984 params)
- Cycle 286 [2] K=512 4k:  **43.563**  (393,984 params, current PPL Pareto winner)
- Float ref:               53.895
- Min 1-block best_mean:   **42.785**
- Delta vs cycle 271 [1] 4k: **-1.035** PPL
- Delta vs cycle 286 [2] 4k: **-0.777** PPL
- Delta vs float:            **-11.110** PPL
