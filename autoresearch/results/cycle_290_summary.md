# Cycle 290 — Item 300: 1-block [2] K=512 extended budget (8000 steps)

**Verdict: INCONCLUSIVE**

- Criteria: CONFIRMED if best_mean ≤ 42.5; REFUTED if best_mean ≥ 43.0; else INCONCLUSIVE
- Subsets: [[2]]
- Config: K=512, LR=0.0001, N_STEPS=8000, EVAL_EVERY=200, SEEDS=[42, 137, 2024]
- Float PPL: 53.895
- Wall time: 12.8 min; fits=3

## Per-subset aggregates

| subset | n_seeds | init_mean | best_mean ± std | best_step | final@8000 |
|---|---|---|---|---|---|
| [2] | 3 | 55.46118 | **42.752 ± 0.032** | 7933 ± 94 | 42.754 |

**Min over subsets: 42.752 (2)**

## Per-seed details

### Subset [2]

| seed | init | best | best_step | final@8000 |
|---|---|---|---|---|
| 42 | 55.46118 | 42.777 | 8000 | 42.777 |
| 137 | 55.46119 | 42.772 | 7800 | 42.779 |
| 2024 | 55.46118 | 42.707 | 8000 | 42.707 |

## Comparison

- Cycle 286 [2] K=512 4k:  **43.563**  (393,984 params)
- Cycle 287 [1] K=512 8k:  **42.785**  (393,984 params, current PPL Pareto winner)
- Cycle 271 [1] K=512 4k:  **43.820**
- Float ref:               53.895
- Min 1-block best_mean:   **42.752**
- Delta vs cycle 286 [2] 4k: **-0.811** PPL  (pure budget gain on block 2)
- Delta vs cycle 287 [1] 8k: **-0.033** PPL  (matched 8k budget [2] vs [1])
- Delta vs cycle 271 [1] 4k: **-1.068** PPL
- Delta vs float:            **-11.143** PPL
