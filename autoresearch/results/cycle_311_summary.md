# Cycle 311 — Item 321: Rank-1 LASER for all concentrator blocks {b2, b3, b10, b11}

**Verdict: CONFIRMED**  (n_pass = 4 / 4 at PPL ≤ 60.0)

- Float PPL: 53.8949 (ref 53.8949)
- Blocks: [2, 3, 10, 11], c_proj.weight shape (3072, 768), HF Conv1D
- Trainable per fit: 1,536 = 1·768 (B) + 768 (bias). A=(U[:, :1]·Σ[:1]) frozen.
- Adam LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024], best-state restore.

## No-retrain r=1 PPL per block (sanity)

| block | PPL | Δfloat | %above | top-1 energy |
|---|---|---|---|---|
| 2 | 72.3808 | +18.486 | +34.30% | 0.0716 |
| 3 | 58.7828 | +4.888 | +9.07% | 0.0837 |
| 10 | 60.6446 | +6.750 | +12.52% | 0.0231 |
| 11 | 60.6103 | +6.715 | +12.46% | 0.0401 |

- b2 r=1 noretrain cross-check vs cycle 285 ref 72.3808: |Δ| = 0.0000

## Retrain at r=1 (3 seeds per block, Adam LR=0.0001, 4000 steps)

| block | seed | init | best | best_step | final@N | restored |
|---|---|---|---|---|---|---|
| 2 | 42 | 72.3808 | 49.6521 | 3700 | 50.3468 | 49.6521 |
| 2 | 137 | 72.3808 | 49.6408 | 3900 | 50.1220 | 49.6408 |
| 2 | 2024 | 72.3808 | 49.9688 | 3300 | 50.0249 | 49.9688 |
| 3 | 42 | 58.7828 | 55.0865 | 3900 | 55.1946 | 55.0865 |
| 3 | 137 | 58.7828 | 55.0434 | 3900 | 55.1397 | 55.0434 |
| 3 | 2024 | 58.7828 | 55.0155 | 4000 | 55.0155 | 55.0155 |
| 10 | 42 | 60.6446 | 52.9358 | 2500 | 53.0183 | 52.9358 |
| 10 | 137 | 60.6446 | 52.8993 | 3100 | 52.9989 | 52.8993 |
| 10 | 2024 | 60.6446 | 52.8971 | 2800 | 53.0055 | 52.8971 |
| 11 | 42 | 60.6103 | 59.7154 | 3600 | 59.7156 | 59.7154 |
| 11 | 137 | 60.6103 | 59.6968 | 3600 | 59.7037 | 59.6968 |
| 11 | 2024 | 60.6103 | 59.7099 | 3900 | 59.7253 | 59.7099 |

## Per-block aggregate

| block | best_mean | best_std | best_min | best_max | passes ≤60 |
|---|---|---|---|---|---|
| 2 | 49.7539 | 0.1520 | 49.6408 | 49.9688 | yes |
| 3 | 55.0485 | 0.0292 | 55.0155 | 55.0865 | yes |
| 10 | 52.9107 | 0.0177 | 52.8971 | 52.9358 | yes |
| 11 | 59.7074 | 0.0078 | 59.6968 | 59.7154 | yes |

## Verdict logic (item 321 verbatim)

- n_pass := |{b ∈ [2, 3, 10, 11] : best_mean ≤ 60.0}| = **4**
- CONFIRMED iff n_pass ≥ 3
- REFUTED iff n_pass ≤ 1
- VERDICT: **CONFIRMED**

## Comparisons

- Cycle 292 b1 r=1 best_mean: 50.0794
- Cycle 280 b1 r=2 best_mean: 49.0324
- Cycle 285 b2 r=2 best_mean: 49.329
- Cycle 288 b3 r=24 best_mean: 50.5935
- Cycle 289 b10 r=5 best_mean: 51.4659
- Cycle 270 b11 r=22 best_mean: 54.4687

| block | r=1 (this cycle) | multi-rank anchor | rank | Δ (r1 − anchor) |
|---|---|---|---|---|
| 2 | 49.7539 | 49.3290 | 2 | +0.4249 |
| 3 | 55.0485 | 50.5935 | 24 | +4.4550 |
| 10 | 52.9107 | 51.4659 | 5 | +1.4448 |
| 11 | 59.7074 | 54.4687 | 22 | +5.2387 |
