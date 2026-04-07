# Cycle 294 — Item 304: 1-block [1] K sweep — 8000 steps

**Verdict: CONFIRMED**

- Criteria: CONFIRMED if ≥2 of 4 K-tiers shift by ≥0.5 PPL; REFUTED if all ≤0.1 PPL; else INCONCLUSIVE
- BLOCKS: [1]; K_VALUES: [32, 64, 128, 256]
- Config: LR=0.0001, N_STEPS=8000, EVAL_EVERY=200, SEEDS=[42, 137, 2024]
- Float PPL: 53.895
- Wall time: 50.6 min; fits=12

## Per-K aggregates (4k → 8k extension)

| K | params | n_seeds | 4k best_mean (cyc 276) | 8k best_mean ± std | shift Δ | best_min/max | best_step | final@8000 |
|---|---|---|---|---|---|---|---|---|
| 32 | 25,344 | 3 | 49.649 | **47.726 ± 0.022** | **+1.923** | 47.694 / 47.745 | 8000 ± 0 | 47.726 |
| 64 | 49,920 | 3 | 48.045 | **45.678 ± 0.011** | **+2.367** | 45.663 / 45.689 | 8000 ± 0 | 45.678 |
| 128 | 99,072 | 3 | 46.932 | **44.656 ± 0.019** | **+2.276** | 44.630 / 44.675 | 8000 ± 0 | 44.656 |
| 256 | 197,376 | 3 | 45.385 | **43.487 ± 0.026** | **+1.899** | 43.453 / 43.518 | 8000 ± 0 | 43.487 |

## Per-seed details

### K=32

| seed | init | best | best_step | final@8000 | step-4000 eval | cyc 276 ref | \|Δ\| |
|---|---|---|---|---|---|---|---|
| 42 | 55.6239 | 47.7445 | 8000 | 47.7445 | 49.623039 | 49.623039 | 0.00e+00 |
| 137 | 55.6239 | 47.6944 | 8000 | 47.6944 | 49.683719 | 49.683719 | 0.00e+00 |
| 2024 | 55.6239 | 47.7385 | 8000 | 47.7385 | 49.638872 | 49.638872 | 0.00e+00 |

### K=64

| seed | init | best | best_step | final@8000 | step-4000 eval | cyc 276 ref | \|Δ\| |
|---|---|---|---|---|---|---|---|
| 42 | 55.5249 | 45.6628 | 8000 | 45.6628 | 48.024281 | 48.024281 | 0.00e+00 |
| 137 | 55.5249 | 45.6829 | 8000 | 45.6829 | 48.100182 | 48.100182 | 0.00e+00 |
| 2024 | 55.5249 | 45.6889 | 8000 | 45.6889 | 48.011290 | 48.011290 | 0.00e+00 |

### K=128

| seed | init | best | best_step | final@8000 | step-4000 eval | cyc 276 ref | \|Δ\| |
|---|---|---|---|---|---|---|---|
| 42 | 55.2727 | 44.6300 | 8000 | 44.6300 | 46.865833 | 46.865833 | 0.00e+00 |
| 137 | 55.2727 | 44.6748 | 8000 | 44.6748 | 47.020823 | 47.020823 | 0.00e+00 |
| 2024 | 55.2727 | 44.6617 | 8000 | 44.6617 | 46.908171 | 46.908171 | 0.00e+00 |

### K=256

| seed | init | best | best_step | final@8000 | step-4000 eval | cyc 276 ref | \|Δ\| |
|---|---|---|---|---|---|---|---|
| 42 | 54.9026 | 43.5178 | 8000 | 43.5178 | 45.403186 | 45.403179 | 6.93e-06 |
| 137 | 54.9026 | 43.4534 | 8000 | 43.4534 | 45.427799 | 45.427796 | 2.92e-06 |
| 2024 | 54.9026 | 43.4883 | 8000 | 43.4883 | 45.325150 | 45.325146 | 4.00e-06 |

## Comparison vs baselines

- Float ref:                      53.895
- Cycle 271 [1] K=512 4k:         43.820 (393,984 params)
- Cycle 287 [1] K=512 8k:         42.785 (393,984 params)

- 1-block K=32 8k best_mean: **47.726**, shift vs cyc 276 4k: **+1.923** PPL, params: 25,344
- 1-block K=64 8k best_mean: **45.678**, shift vs cyc 276 4k: **+2.367** PPL, params: 49,920
- 1-block K=128 8k best_mean: **44.656**, shift vs cyc 276 4k: **+2.276** PPL, params: 99,072
- 1-block K=256 8k best_mean: **43.487**, shift vs cyc 276 4k: **+1.899** PPL, params: 197,376

## Cross-checks

All 12 step-4000 evals in cycle 294 match cycle 276 best_ppl per-(K,seed) within |Δ|≤6.93e-06
(MPS nondeterminism noise, well within 1e-4 tolerance). The entire 4k→8k PPL gain is unambiguously
attributed to the extra 4000 Adam steps, not to init or LR drift.

## Reading

All 4 K-tiers gain **1.90–2.37 PPL** from the budget extension:

- K=32 (25K params): +1.923 (49.649 → 47.726)
- K=64 (50K params): +2.367 (48.045 → 45.678)
- K=128 (99K params): +2.276 (46.932 → 44.656)
- K=256 (197K params): +1.899 (45.385 → 43.487)

All 12 fits (3 seeds × 4 K) are BUDGET-BOUNDED at step 8000 — the floor is not yet found for any K.
All shifts far exceed the 0.5 PPL threshold (item 304 requires ≥2 of 4; 4/4 clear it).

The gain is larger than the [1] K=512 4k→8k extension of cycle 287 (−1.04 PPL), falsifying the
brief's 'fewer params → larger gain from Adam slack' framing in the direction of magnitude — smaller
K at [1] gains MORE than K=512 did, consistent with Adam slack being larger when trainable parameter
count is smaller. K=32 gains +1.92 PPL (16x fewer params than K=512 which gained +1.04).
K=64 and K=128 gain the most (~2.3 PPL) — they are the mid-tier sweet spot under 8k Adam.

Comparison to cycle 287 [1] K=512 8k = 42.785 (394K params):
- cycle 294 K=256 8k = 43.487 (197K params) — 0.70 PPL WORSE at 2× fewer params.
- cycle 294 K=128 8k = 44.656 (99K params) — 1.87 PPL WORSE at 4× fewer params.
- cycle 294 K=64 8k = 45.678 (50K params) — 2.89 PPL WORSE at 8× fewer params.
- cycle 294 K=32 8k = 47.726 (25K params) — 4.94 PPL WORSE at 16× fewer params.

The K=256 8k point (197K, 43.49) is a new Pareto anchor — at half the params of cycle 287, it's only
0.70 PPL worse. This updates the 1-block [1] channel-K Pareto curve with two new frontier points.
