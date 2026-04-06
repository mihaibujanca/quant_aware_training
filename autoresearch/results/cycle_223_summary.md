# Cycle 223 — Item 228: Lower K Adam correction

**Verdict: CONFIRMED**

- Criteria (K=256): Confirmed ≤52.0, Refuted >58.0, Inconclusive 52-58
- Config: BLOCKS=[1, 4, 7], K_VALUES=[256, 128], LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895
- K=512 baseline (cycle 210): 46.556

## Compression efficiency curve

| K | trainable (3 blocks) | best_ppl_mean | std | %vs float |
|---|---|---|---|---|
| 128 | 297,216 | 48.989 | 0.033 | -9.10% |
| 256 | 592,128 | 47.509 | 0.023 | -11.85% |
| 512 (c210) | 1,181,952 | 46.556 | — | -13.62% |

## Per-K aggregates

### K = 256

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 63.655 | 47.516 | 3900 | 47.523 |
| 137 | 63.655 | 47.478 | 4000 | 47.478 |
| 2024 | 63.655 | 47.533 | 4000 | 47.533 |
| **mean** | 63.655 | **47.509** | 3967 | 47.511 |
| **std**  | 0.000 | 0.023 | 47 | 0.024 |

### K = 128

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 65.098 | 48.987 | 3900 | 48.989 |
| 137 | 65.098 | 48.950 | 4000 | 48.950 |
| 2024 | 65.098 | 49.030 | 4000 | 49.030 |
| **mean** | 65.098 | **48.989** | 3967 | 48.990 |
| **std**  | 0.000 | 0.033 | 47 | 0.033 |

## Comparison to K=512 (cycle 210)

- K=512 best_mean (cycle 210): **46.556** (1,181,952 trainable across 3 blocks)
- K=256 best_mean (this cycle): **47.509** (delta vs K=512: +0.953)
- K=128 best_mean (this cycle): **48.989** (delta vs K=512: +2.433)

