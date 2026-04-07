# Cycle 282 — Item 292: Targeted block 10 low-rank c_proj compression

**Verdict: CONFIRMED**

- Float PPL: 53.8949 (ref 53.895)
- Block: 10, c_proj.weight shape (3072, 768), HF Conv1D
- Cycle 278 reference: eff_rank(c_proj_10 OUTPUT) = 16.7 (top-1 energy 0.5889)

## No-retrain rank-r sweep

| r | PPL | Δfloat | %above | top-r energy |
|---|---|---|---|---|
| 5 | 59.9179 | +6.023 | +11.18% | 0.0503 |
| 17 | 59.3783 | +5.483 | +10.17% | 0.1024 |
| 50 | 58.2649 | +4.370 | +8.11% | 0.2118 |

## Retrain at r=17 (3 seeds, Adam LR=0.0001, 4000 steps)

Trainable params: 13,824 = 17×768 (B) + 768 (bias). A=(U[:, :17]·Σ[:17]) frozen.

| seed | init | best | best_step | final@N | restored |
|---|---|---|---|---|---|
| 42 | 59.3783 | 50.0048 | 3900 | 50.0948 | 50.0048 |
| 137 | 59.3783 | 50.0042 | 3900 | 50.0650 | 50.0042 |
| 2024 | 59.3783 | 50.1245 | 3800 | 50.1536 | 50.1245 |
| **mean** | — | **50.0445** | — | — | — |
| **std**  | — | 0.0565 | — | — | — |

## Verdict logic (item 292 verbatim)

- r=17 retrain best_mean = 50.0445; CONFIRMED iff ≤ 55.0
- all r ∈ [5, 17, 50] no-retrain ≥ 57.0? True; REFUTED iff ALL no-retrain AND retrain ≥ 57.0
- VERDICT: **CONFIRMED**

## Comparisons

- Δ vs float: noretrain r=17 = +5.483; retrain best_mean = -3.850
- Δ retrain vs noretrain at r=17: -9.334 (Adam recovery)
- Δ vs cycle 270 block 11 rank-22 retrain (54.4687): -4.424 PPL
- Δ vs cycle 280 block 1 rank-2 retrain (49.0324): +1.012 PPL
- Trainable param ratio vs cycle 270 (17,664): 0.7826 (cycle 282=13,824)
- Trainable param ratio vs cycle 280 (2,304): 6.0000
