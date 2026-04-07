# Cycle 283 — Item 293: 1-block [2] K sweep

**Verdict: CONFIRMED**

- Criteria: CONFIRMED iff K=16 best_mean ≤ 52.0 OR K=32 best_mean ≤ 50.0; REFUTED iff all K ≥ 55.0; else INCONCLUSIVE
- BLOCKS: [2]; K_VALUES: [16, 32, 64]
- Config: LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895
- Wall time: 22.6 min; fits=9

## Per-K aggregates

| K | params | n_seeds | init_mean | best_mean ± std | best_min/max | best_step | final@4000 |
|---|---|---|---|---|---|---|---|
| 16 | 13,056 | 3 | 57.495 | **49.462 ± 0.047** | 49.417 / 49.527 | 3933 ± 94 | 49.479 |
| 32 | 25,344 | 3 | 56.590 | **47.897 ± 0.066** | 47.833 / 47.987 | 4000 ± 0 | 47.897 |
| 64 | 49,920 | 3 | 56.473 | **47.511 ± 0.061** | 47.440 / 47.589 | 4000 ± 0 | 47.511 |

## Per-seed details

### K=16

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 57.495 | 49.417 | 4000 | 49.417 |
| 137 | 57.495 | 49.527 | 3800 | 49.576 |
| 2024 | 57.495 | 49.442 | 4000 | 49.442 |

### K=32

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 56.590 | 47.833 | 4000 | 47.833 |
| 137 | 56.590 | 47.987 | 4000 | 47.987 |
| 2024 | 56.590 | 47.870 | 4000 | 47.870 |

### K=64

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 56.473 | 47.440 | 4000 | 47.440 |
| 137 | 56.473 | 47.589 | 4000 | 47.589 |
| 2024 | 56.473 | 47.506 | 4000 | 47.506 |

## Comparison

- Cycle 271 [1] K=512 (394K):             **43.820**
- Cycle 281 [1,2] K=512 2-block (788K):  44.339
- Cycle 276 [1] K=32  (25K):              49.649
- Cycle 276 [1] K=64  (50K):              48.045
- Cycle 276 [1] K=128 (99K):              46.932
- Cycle 276 [1] K=256 (197K):             45.385
- Float ref:                               53.895

- 1-block [2] K=16 best_mean: **49.462**, Δ vs float: **-4.433** PPL
- 1-block [2] K=32 best_mean: **47.897**, Δ vs float: **-5.998** PPL
- 1-block [2] K=64 best_mean: **47.511**, Δ vs float: **-6.384** PPL
