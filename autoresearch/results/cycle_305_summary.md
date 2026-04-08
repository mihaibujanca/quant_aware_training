# Cycle 305 — Item 315: Multi-block LASER stacking [1, 2, 3, 10, 11]

**Verdict: REFUTED**

- Float PPL: 53.8949 (ref 53.895)
- Blocks: [1, 2, 3, 10, 11], ranks: [2, 2, 24, 5, 22]
- Trainable params per fit: 46,080 = sum_l (r_l*768 + 768)

## Joint Adam retrain (3 seeds, LR=0.0001, 4000 steps)

| seed | init | best | best_step | final@N | restored |
|---|---|---|---|---|---|
| 42 | 150.3568 | 72.8047 | 3500 | 74.4899 | 72.8047 |
| 137 | 150.3568 | 73.1217 | 3700 | 73.7349 | 73.1217 |
| 2024 | 150.3568 | 73.0915 | 4000 | 73.0915 | 73.0915 |
| **mean** | 150.3568 | **73.0060** | — | 73.7721 | — |
| **std**  | — | 0.1429 | — | — | — |

## Verdict logic (item 315 verbatim)

- best_mean = 73.0060
- CONFIRMED iff best_mean ≤ 51.0? False
- REFUTED iff best_mean ≥ 55.0? True
- VERDICT: **REFUTED**

## Comparisons

- Δ vs float (53.895): +19.111
- Δ vs b1 r=2 (49.0324): +23.974
- Δ vs b2 r=2 (49.329): +23.677
- Δ vs b3 r=24 (50.5935): +22.412
- Δ vs b10 r=5 (51.4659): +21.540
- Δ vs b11 r=22 (54.4687): +18.537
- Δ vs cycle 284 hybrid b1+b11 (52.8803): +20.126
- Δ vs cycle 291 hybrid b1+b10 (51.8008): +21.205
