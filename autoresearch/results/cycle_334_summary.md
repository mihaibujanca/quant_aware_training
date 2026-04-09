# Cycle 334 — Item 328: Extended budget rank-1 retrain at b11 (8000 steps)

**Verdict: INCONCLUSIVE**

- Block: 11, rank: 1, steps: 8000
- LR schedule: 0.0001 (steps 1-4000) → 5e-05 (steps 4001-8000)
- Float PPL: 53.8949 (ref 53.8949)
- Init PPL (no-retrain r=1): 60.6103 (cycle 311 ref: 60.6103)
- Trainable: 1536 params (B=1×768 + bias)
- Seeds: [42, 137, 2024], best-state restore

## Per-seed results

| seed | init | best | best_step | best@4000 | Δ vs c311 | final@8000 |
|---|---|---|---|---|---|---|
| 42 | 60.6103 | 59.5762 | 7800 | 59.7154 | +0.0000 | 59.5802 |
| 137 | 60.6103 | 59.5466 | 8000 | 59.6968 | +0.0000 | 59.5466 |
| 2024 | 60.6103 | 59.5703 | 7900 | 59.7099 | +0.0000 | 59.5756 |

## Aggregate

- best_mean_ppl (8000 steps): **59.5644** ± 0.0128
- best_mean_ppl_at_4000: 59.7074 (cycle 311 ref: 59.7074)

## Verdict logic

- CONFIRMED-robust-r1 iff best_mean_ppl ≤ 58.0
- REFUTED-borderline-artifact iff best_mean_ppl ≥ 60.0
- best_mean_ppl = 59.5644 → **INCONCLUSIVE**

## Last 10 eval points per seed (plateau analysis)

### seed 42
| step | eval_ppl |
|---|---|
| 7100 | 59.6312 |
| 7200 | 59.6096 |
| 7300 | 59.6282 |
| 7400 | 59.6041 |
| 7500 | 59.6230 |
| 7600 | 59.6072 |
| 7700 | 59.6061 |
| 7800 | 59.5762 |
| 7900 | 59.5904 |
| 8000 | 59.5802 |

### seed 137
| step | eval_ppl |
|---|---|
| 7100 | 59.5749 |
| 7200 | 59.5715 |
| 7300 | 59.5595 |
| 7400 | 59.5655 |
| 7500 | 59.5678 |
| 7600 | 59.5639 |
| 7700 | 59.5684 |
| 7800 | 59.5701 |
| 7900 | 59.5579 |
| 8000 | 59.5466 |

### seed 2024
| step | eval_ppl |
|---|---|
| 7100 | 59.5907 |
| 7200 | 59.5914 |
| 7300 | 59.5840 |
| 7400 | 59.5884 |
| 7500 | 59.5978 |
| 7600 | 59.5948 |
| 7700 | 59.5926 |
| 7800 | 59.5829 |
| 7900 | 59.5703 |
| 8000 | 59.5756 |
