# Cycle 270 — Item 280: Targeted block 11 low-rank c_proj compression

**Verdict: CONFIRMED**

- Float PPL: 53.8949 (ref 53.895)
- Block: 11, c_proj.weight shape (3072, 768), HF Conv1D
- SVD cycle 267 reference: eff_rank(c_proj_11 OUTPUT) = 22.49

## No-retrain rank-r sweep

| r | PPL | Δfloat | %above | top-r energy |
|---|---|---|---|---|
| 5 | 60.9913 | +7.096 | +13.17% | 0.0745 |
| 22 | 59.6834 | +5.789 | +10.74% | 0.1414 |
| 50 | 58.6337 | +4.739 | +8.79% | 0.2249 |
| 100 | 57.3055 | +3.411 | +6.33% | 0.3483 |

## Retrain at r=22 (3 seeds, Adam LR=0.0001, 4000 steps)

Trainable params: 17,664 = 22×768 (B) + 768 (bias). A=(U[:, :22]·Σ[:22]) frozen.

| seed | init | best | best_step | final@N | restored |
|---|---|---|---|---|---|
| 42 | 59.6834 | 54.4673 | 3800 | 54.4764 | 54.4673 |
| 137 | 59.6834 | 54.4405 | 3600 | 54.4765 | 54.4405 |
| 2024 | 59.6834 | 54.4984 | 3900 | 54.5440 | 54.4984 |
| **mean** | — | **54.4687** | — | — | — |
| **std**  | — | 0.0237 | — | — | — |

## Verdict logic (item 280 verbatim)

- rank-22 no-retrain PPL = 59.6834; CONFIRMED iff ≤ 55.0; REFUTED iff ≥ 60.0
- rank-22 retrain best_mean = 54.4687; CONFIRMED iff ≤ 54.5; REFUTED iff ≥ 58.0
- VERDICT: **CONFIRMED**

## Comparisons

- Δ vs float: noretrain r=22 = +5.789; retrain best_mean = +0.574
- Δ retrain vs noretrain at r=22: -5.215 (Adam recovery)
