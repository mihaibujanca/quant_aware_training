# Cycle 288 — Item 298: Targeted block 3 low-rank c_proj compression

**Verdict: CONFIRMED**

- Float PPL: 53.8949 (ref 53.895)
- Block: 3, c_proj.weight shape (3072, 768), HF Conv1D
- Cycle 278 reference: eff_rank(c_proj_3 OUTPUT) = 24.2853 (top-1 energy 0.5887)

## No-retrain rank-r sweep

| r | PPL | Δfloat | %above | top-r energy |
|---|---|---|---|---|
| 5 | 58.4581 | +4.563 | +8.47% | 0.1136 |
| 10 | 58.3169 | +4.422 | +8.20% | 0.1423 |
| 24 | 57.2062 | +3.311 | +6.14% | 0.2067 |

## Retrain at r=24 (3 seeds, Adam LR=0.0001, 4000 steps)

Trainable params: 19,200 = 24×768 (B) + 768 (bias). A=(U[:, :24]·Σ[:24]) frozen.

| seed | init | best | best_step | final@N | restored |
|---|---|---|---|---|---|
| 42 | 57.2062 | 50.6474 | 3900 | 50.7107 | 50.6474 |
| 137 | 57.2062 | 50.5825 | 3900 | 50.6320 | 50.5825 |
| 2024 | 57.2062 | 50.5507 | 4000 | 50.5507 | 50.5507 |
| **mean** | — | **50.5935** | — | — | — |
| **std**  | — | 0.0402 | — | — | — |

## Verdict logic (item 298 verbatim)

- r=24 retrain best_mean = 50.5935; CONFIRMED iff ≤ 55.0; REFUTED iff ≥ 57.0
- VERDICT: **CONFIRMED**

## Comparisons

- Δ vs float: noretrain r=24 = +3.311; retrain best_mean = -3.301
- Δ retrain vs noretrain at r=24: -6.613 (Adam recovery)
- Δ vs cycle 282 block 10 rank-17 retrain (50.0445): +0.549 PPL
- Δ vs cycle 270 block 11 rank-22 retrain (54.4687): -3.875 PPL
- Δ vs cycle 280 block 1 rank-2 retrain (49.0324): +1.561 PPL
- Δ vs cycle 285 block 2 rank-2 retrain (49.329): +1.265 PPL
- Trainable param ratio vs cycle 270 (17,664): 1.0870 (cycle 288=19,200)
- Trainable param ratio vs cycle 282 (13,824): 1.3889
