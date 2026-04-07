# Cycle 293 — Item 303: Targeted block 0 low-rank c_proj compression

**Verdict: REFUTED**

- Float PPL: 53.8949 (ref 53.895)
- Block: 0, c_proj.weight shape (3072, 768), HF Conv1D
- Cycle 278 reference: eff_rank(c_proj_0 OUTPUT) = 71.93

## No-retrain rank-r sweep

| r | PPL | Δfloat | %above | top-r energy |
|---|---|---|---|---|
| 10 | 1939.0057 | +1885.111 | +3497.75% | 0.1495 |
| 30 | 913.7012 | +859.806 | +1595.34% | 0.2241 |
| 72 | 396.2582 | +342.363 | +635.24% | 0.3271 |

## Retrain at r=72 (3 seeds, Adam LR=0.0001, 4000 steps)

Trainable params: 56,064 = 72×768 (B) + 768 (bias). A=(U[:, :72]·Σ[:72]) frozen.

| seed | init | best | best_step | final@N | restored |
|---|---|---|---|---|---|
| 42 | 396.2582 | 63.0582 | 4000 | 63.0582 | 63.0582 |
| 137 | 396.2582 | 63.0645 | 4000 | 63.0645 | 63.0645 |
| 2024 | 396.2582 | 63.2752 | 3900 | 63.5582 | 63.2752 |
| **mean** | — | **63.1327** | — | — | — |
| **std**  | — | 0.1009 | — | — | — |

## Verdict logic (item 303 verbatim)

- r=72 retrain best_mean = 63.1327; CONFIRMED iff ≤ 55.0; REFUTED iff ≥ 58.0
- VERDICT: **REFUTED**

## Comparisons

- Δ vs float: noretrain r=72 = +342.363; retrain best_mean = +9.238
- Δ retrain vs noretrain at r=72: -333.126 (Adam recovery)
- Δ vs cycle 282 block 10 rank-17 retrain (50.0445): +13.088 PPL
- Δ vs cycle 270 block 11 rank-22 retrain (54.4687): +8.664 PPL
- Δ vs cycle 280 block 1 rank-2 retrain (49.0324): +14.100 PPL
- Δ vs cycle 285 block 2 rank-2 retrain (49.329): +13.804 PPL
- Δ vs cycle 288 block 3 rank-24 retrain (50.5935): +12.539 PPL
- Δ vs cycle 292 block 1 rank-1 retrain (50.0794): +13.053 PPL
- Trainable param ratio vs cycle 270 (17,664): 3.1739 (cycle 293=56,064)
- Trainable param ratio vs cycle 288 (19,200): 2.9200
