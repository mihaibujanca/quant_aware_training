# Cycle 292 — Item 302: Block 1 rank-1 c_proj sub-eff_rank stress test

**Verdict: CONFIRMED**

- Float PPL: 53.8949 (ref 53.895)
- Block: 1, c_proj.weight shape (3072, 768), HF Conv1D
- Cycle 278 reference: eff_rank(c_proj_1 OUTPUT) = 1.9448 (top-1 energy 0.9294)
- Sub-eff_rank test: r=1 < eff_rank 1.9448 (r/eff_rank = 0.514)

## No-retrain rank-r sweep

| r | PPL | Δfloat | %above | top-r energy |
|---|---|---|---|---|
| 1 | 59.7513 | +5.856 | +10.87% | 0.0459 |
- Cycle 280 reported r=1 no-retrain = 59.7513, measured here = 59.7513, |Δ| = 0.0000

## Retrain at r=1 (3 seeds, Adam LR=0.0001, 4000 steps)

Trainable params: 1,536 = 1×768 (B) + 768 (bias). A=(U[:, :1]·Σ[:1]) frozen.

| seed | init | best | best_step | final@N | restored |
|---|---|---|---|---|---|
| 42 | 59.7513 | 50.1471 | 3900 | 50.2724 | 50.1471 |
| 137 | 59.7513 | 50.0468 | 3900 | 50.1260 | 50.0468 |
| 2024 | 59.7513 | 50.0444 | 4000 | 50.0444 | 50.0444 |
| **mean** | — | **50.0794** | — | — | — |
| **std**  | — | 0.0479 | — | — | — |

## Verdict logic (item 302 verbatim)

- r=1 retrain best_mean = 50.0794
- CONFIRMED iff ≤ 51.0
- REFUTED iff ≥ 53.5
- VERDICT: **CONFIRMED**

## Comparisons

- Δ vs float: noretrain r=1 = +5.856; retrain best_mean = -3.815
- Δ retrain vs noretrain at r=1: -9.672 (Adam recovery)
- Δ vs cycle 280 block 1 r=2 (49.0324, 2,304 params): +1.047 PPL (2× more params)
- Δ vs cycle 289 block 10 r=5 (51.4659, 4,608 params): -1.386 PPL (3× more params)
- Δ vs cycle 285 block 2 r=2 (49.329, 2,304 params): +0.750 PPL
- Δ vs cycle 282 block 10 r=17 (50.0445, 13,824 params): +0.035 PPL
- Δ vs cycle 270 block 11 r=22 (54.4687, 17,664 params): -4.389 PPL
- Δ vs cycle 288 block 3 r=24 (50.5935, 19,200 params): -0.514 PPL
- Trainable param ratio vs cycle 280 (2,304): 0.6667 (cycle 292 = 1,536)
- Trainable param ratio vs cycle 289 (4,608): 0.3333
