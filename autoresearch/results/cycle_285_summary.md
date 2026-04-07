# Cycle 285 — Item 295: Targeted block 2 low-rank c_proj compression

**Verdict: CONFIRMED**

- Float PPL: 53.8949 (ref 53.895)
- Block: 2, c_proj.weight shape (3072, 768), HF Conv1D
- Cycle 278 reference: eff_rank(c_proj_2 OUTPUT) = 1.0503 (top-1 energy 0.9961) — model minimum

## No-retrain rank-r sweep

| r | PPL | Δfloat | %above | top-r energy |
|---|---|---|---|---|
| 1 | 72.3808 | +18.486 | +34.30% | 0.0716 |
| 2 | 64.0814 | +10.186 | +18.90% | 0.0828 |
| 5 | 64.0945 | +10.200 | +18.92% | 0.1049 |
| 10 | 62.1562 | +8.261 | +15.33% | 0.1342 |

## Retrain at r=2 (3 seeds, Adam LR=0.0001, 4000 steps)

Trainable params: 2,304 = 2×768 (B) + 768 (bias). A=(U[:, :2]·Σ[:2]) frozen.

| seed | init | best | best_step | final@N | restored |
|---|---|---|---|---|---|
| 42 | 64.0814 | 49.2446 | 3700 | 49.9992 | 49.2446 |
| 137 | 64.0814 | 49.2253 | 3900 | 49.7387 | 49.2253 |
| 2024 | 64.0814 | 49.5171 | 3300 | 49.5760 | 49.5171 |
| **mean** | — | **49.3290** | — | — | — |
| **std**  | — | 0.1332 | — | — | — |

## Retrain at r=1 — SKIPPED (noretrain r=1 PPL = 72.3808 > gate 55.0)

## Verdict logic (item 295 verbatim)

- r=2 retrain best_mean = 49.3290; CONFIRMED iff ≤ 50.0; REFUTED iff ≥ 53.0
- r=1 retrain SKIPPED
- VERDICT: **CONFIRMED**

## Comparisons

- Δ vs float: noretrain r=2 = +10.186; retrain r=2 best_mean = -4.566
- Δ retrain vs noretrain at r=2: -14.752 (Adam recovery)
- Δ vs cycle 280 b1 r=2 (2,304 params, 49.0324): +0.297
- Δ vs cycle 270 b11 r=22 (17,664 params, 54.4687): -5.140
- Δ vs cycle 282 b10 r=17 (13,824 params, 50.0445): -0.716
- Δ vs cycle 283 [2] K=16 (13,056 params, 49.4621): -0.133
