# Cycle 280 — Item 290: Targeted block 1 low-rank c_proj compression

**Verdict: CONFIRMED**

- Float PPL: 53.8949 (ref 53.895)
- Block: 1, c_proj.weight shape (3072, 768), HF Conv1D
- Cycle 278 reference: eff_rank(c_proj_1 OUTPUT) = 1.9448 (top-1 energy 0.9294)

## No-retrain rank-r sweep

| r | PPL | Δfloat | %above | top-r energy |
|---|---|---|---|---|
| 1 | 59.7513 | +5.856 | +10.87% | 0.0459 |
| 2 | 57.6581 | +3.763 | +6.98% | 0.0757 |
| 5 | 56.8326 | +2.938 | +5.45% | 0.1167 |
| 10 | 56.7302 | +2.835 | +5.26% | 0.1523 |

## Retrain at r=2 (3 seeds, Adam LR=0.0001, 4000 steps)

Trainable params: 2,304 = 2×768 (B) + 768 (bias). A=(U[:, :2]·Σ[:2]) frozen.

| seed | init | best | best_step | final@N | restored |
|---|---|---|---|---|---|
| 42 | 57.6581 | 49.0603 | 3900 | 49.2498 | 49.0603 |
| 137 | 57.6581 | 49.0367 | 3900 | 49.0741 | 49.0367 |
| 2024 | 57.6581 | 49.0004 | 4000 | 49.0004 | 49.0004 |
| **mean** | — | **49.0324** | — | — | — |
| **std**  | — | 0.0246 | — | — | — |

## Verdict logic (item 290 verbatim)

- any r ∈ [1, 2, 5, 10] no-retrain ≤ 55.0? False
- r=2 retrain best_mean = 49.0324; CONFIRMED iff ≤ 54.0; REFUTED iff ≥ 55.0
- all r no-retrain ≥ 60.0? False
- VERDICT: **CONFIRMED**

## Comparisons

- Δ vs float: noretrain r=2 = +3.763; retrain best_mean = -4.862
- Δ retrain vs noretrain at r=2: -8.626 (Adam recovery)
- Δ vs cycle 270 block 11 rank-22 retrain (54.4687): -5.436
- Trainable param ratio vs cycle 270: 0.1304 (cycle 270=17,664; cycle 280=2,304)
