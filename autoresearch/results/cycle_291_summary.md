# Cycle 291 — Item 301: Hybrid block 1 rank-2 + block 10 rank-5 c_proj

**Verdict: REFUTED**

- Float PPL: 53.8949 (ref 53.895)
- Blocks: [1, 10], ranks: [2, 5]
- Trainable params per fit: 6,912 = (2·768 + 768) + (5·768 + 768) = 2,304 + 4,608

## Joint Adam retrain (3 seeds, LR=0.0001, 4000 steps)

| seed | init | best | best_step | final@N | restored |
|---|---|---|---|---|---|
| 42 | 64.4650 | 51.7511 | 3900 | 52.0684 | 51.7511 |
| 137 | 64.4650 | 51.8509 | 3800 | 51.8570 | 51.8509 |
| 2024 | 64.4650 | 51.8004 | 4000 | 51.8004 | 51.8004 |
| **mean** | 64.4650 | **51.8008** | — | 51.9086 | — |
| **std**  | — | 0.0407 | — | — | — |

## Verdict logic (item 301 verbatim)

- best_mean = 51.8008
- CONFIRMED iff best_mean ≤ 49.5? False
- REFUTED iff best_mean ≥ 51.5? True
- VERDICT: **REFUTED**

## Comparisons

- Δ vs float (53.895): -2.094
- Δ vs cycle 280 b1 r=2 (49.0324): +2.768
- Δ vs cycle 289 b10 r=5 (51.4659): +0.335
- Δ vs cycle 282 b10 r=17 (50.0445): +1.756
- Δ vs cycle 284 hybrid b1r2+b11r22 (52.88): -1.079
