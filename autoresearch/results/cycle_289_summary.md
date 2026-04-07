# Cycle 289 — Item 299: Block 10 rank-5 c_proj sub-eff_rank stress test

**Verdict: CONFIRMED**

- Float PPL: 53.8949 (ref 53.895)
- Block: 10, c_proj.weight shape (3072, 768), HF Conv1D
- Cycle 278 reference: eff_rank(c_proj_10 OUTPUT) = 16.7 (top-1 energy 0.5889)
- Sub-eff_rank test: r=5 << eff_rank 16.7

## No-retrain rank-r sweep

| r | PPL | Δfloat | %above | top-r energy |
|---|---|---|---|---|
| 5 | 59.9179 | +6.023 | +11.18% | 0.0503 |
- Cycle 282 reported r=5 no-retrain = 59.92, measured here = 59.918, |Δ| = 0.002

## Retrain at r=5 (3 seeds, Adam LR=0.0001, 4000 steps)

Trainable params: 4,608 = 5×768 (B) + 768 (bias). A=(U[:, :5]·Σ[:5]) frozen.

| seed | init | best | best_step | final@N | restored |
|---|---|---|---|---|---|
| 42 | 59.9179 | 51.4783 | 1500 | 51.6569 | 51.4783 |
| 137 | 59.9179 | 51.4614 | 1700 | 51.6385 | 51.4614 |
| 2024 | 59.9179 | 51.4579 | 1800 | 51.6946 | 51.4579 |
| **mean** | — | **51.4659** | — | — | — |
| **std**  | — | 0.0089 | — | — | — |

## Verdict logic (item 299 verbatim)

- r=5 retrain best_mean = 51.4659
- CONFIRMED iff ≤ 53.0
- REFUTED iff ≥ 56.0
- VERDICT: **CONFIRMED**

## Comparisons

- Δ vs float: noretrain r=5 = +6.023; retrain best_mean = -2.429
- Δ retrain vs noretrain at r=5: -8.452 (Adam recovery)
- Δ vs cycle 282 block 10 r=17 (50.0445, 13,824 params): +1.421 PPL (3× fewer params)
- Δ vs cycle 280 block 1 r=2 (49.0324, 2,304 params): +2.433 PPL (2× more params)
- Δ vs cycle 285 block 2 r=2 (49.329, 2,304 params): +2.137 PPL
- Δ vs cycle 270 block 11 r=22 (54.4687, 17,664 params): -3.003 PPL
- Δ vs cycle 288 block 3 r=24 (50.5935, 19,200 params): +0.872 PPL
- Trainable param ratio vs cycle 282 (13,824): 0.3333 (cycle 289 = 4,608)
- Trainable param ratio vs cycle 280 (2,304): 2.0000
