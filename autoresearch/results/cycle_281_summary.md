# Cycle 281 — Item 291: 2-block [1, 2] K=512

**Verdict: INCONCLUSIVE**

- Criteria: CONFIRMED iff best_mean ≤ 43.9; REFUTED iff best_mean ≥ 44.5; else INCONCLUSIVE
- Subset: [1, 2]
- Config: K=512, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895
- Wall time: 7.6 min; fits=3

## Aggregate

| subset | n_seeds | init_mean | best_mean ± std | best_step | final@4000 |
|---|---|---|---|---|---|
| [1, 2] | 3 | 57.207 | **44.3389 ± 0.0070** | 3300 ± 374 | 44.4636 |

## Per-seed details

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 57.2068 | 44.3473 | 3700 | 44.4932 |
| 137 | 57.2068 | 44.3302 | 2800 | 44.4258 |
| 2024 | 57.2068 | 44.3392 | 3400 | 44.4718 |

## Comparison

- Cycle 266 [1, 4] K=512 (2-block, W_out): **43.8525**
- Cycle 271 [1]    K=512 (1-block, W_out): 43.8205
- Cycle 280 block 1 rank-2 c_proj retrain: 49.0324
- Float ref:                                 53.8950
- Cycle 281 [1, 2] best_mean:               **44.3389**
- Δ vs cycle 266 [1, 4]: **+0.4864** PPL
- Δ vs cycle 271 [1]:    **+0.5184** PPL
- Δ vs float:            **-9.5560** PPL
