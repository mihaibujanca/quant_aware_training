# Cycle 284 — Item 294: Hybrid block 1 rank-2 + block 11 rank-22 c_proj

**Verdict: CONFIRMED**

- Float PPL: 53.8949 (ref 53.895)
- Blocks: [1, 11], ranks: [2, 22]
- Trainable params per fit: 19,968 = (2·768 + 768) + (22·768 + 768) = 2,304 + 17,664

## Joint Adam retrain (3 seeds, LR=0.0001, 4000 steps)

| seed | init | best | best_step | final@N | restored |
|---|---|---|---|---|---|
| 42 | 63.4629 | 52.9176 | 3900 | 53.0446 | 52.9176 |
| 137 | 63.4629 | 52.8446 | 3700 | 52.8729 | 52.8446 |
| 2024 | 63.4629 | 52.8787 | 3600 | 52.9043 | 52.8787 |
| **mean** | 63.4629 | **52.8803** | — | 52.9406 | — |
| **std**  | — | 0.0298 | — | — | — |

## Verdict logic (item 294 verbatim)

- best_mean = 52.8803
- CONFIRMED iff best_mean ≤ 55.0? True
- REFUTED iff best_mean ≥ 58.0? False
- VERDICT: **CONFIRMED**

## Comparisons

- Δ vs float (53.895): -1.015
- Δ vs cycle 270 b11 r=22 (54.4687): -1.588
- Δ vs cycle 280 b1 r=2 (49.0324): +3.848
- Δ vs cycle 282 b10 r=17 (50.0445): +2.836
- Δ vs cycle 275 heterogeneous hybrid (48.2004): +4.680 (different scoring axis)
