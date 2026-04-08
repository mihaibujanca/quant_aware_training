# Cycle 314 — Item 324: Rank-1 pairwise LASER composition matrix

**Verdict: CONFIRMED** (CONFIRMED-persistent-destruction)

- (b10, b11) r=1 delta_vs_worst_single_r1 = **+10.3072**  (best_mean 70.0146 vs worst_single_r1 59.7074)
- n_destructive_at_5 (all pairs) = 2 / 10
- n_destructive_other (excluding b10,b11) = 1 / 9

- Float PPL: 53.8949 (ref 53.895)
- Block order: [1, 2, 3, 10, 11]
- All blocks at rank-1; trainable per pair = 3,072 = 2 × (1·768 + 768)
- N_STEPS=2000, LR=0.0001, SEEDS=[42], EVAL_EVERY=100
- Pair sweep wall: 13.2 min

## 5×5 pair PPL matrix (diagonal = cycle 292/311 r=1 anchors)

| block | 1 | 2 | 3 | 10 | 11 |
|---|---|---|---|---|---|
| **1** | *50.0794* | 55.9938 | 56.0217 | 53.5124 | 59.8573 |
| **2** | 55.9938 | *49.7539* | 55.6467 | 54.2429 | 58.2418 |
| **3** | 56.0217 | 55.6467 | *55.0485* | 55.4499 | 62.4403 |
| **10** | 53.5124 | 54.2429 | 55.4499 | *52.9107* | 70.0146 |
| **11** | 59.8573 | 58.2418 | 62.4403 | 70.0146 | *59.7074* |

## Per-pair detail (sorted by delta_vs_worst_single_r1)

| pair | init | best | delta_vs_worst_r1 | delta_vs_best_r1 | destructive | cycle310 ref | delta_314-310 |
|---|---|---|---|---|---|---|---|
| (2,11) | 82.3399 | 58.2418 | -1.4656 | +8.4879 | no | 53.8455 | +4.3963 |
| (1,11) | 66.7997 | 59.8573 | +0.1499 | +9.7779 | no | 53.5812 | +6.2761 |
| (3,10) | 66.8140 | 55.4499 | +0.4014 | +2.5392 | no | 52.2278 | +3.2221 |
| (2,3) | 100.1877 | 55.6467 | +0.5982 | +5.8928 | no | 52.3641 | +3.2826 |
| (1,10) | 69.2294 | 53.5124 | +0.6017 | +3.4330 | no | 52.1901 | +1.3223 |
| (1,3) | 78.4925 | 56.0217 | +0.9732 | +5.9423 | no | 51.6544 | +4.3673 |
| (2,10) | 82.7204 | 54.2429 | +1.3322 | +4.4890 | no | 52.9059 | +1.3370 |
| (3,11) | 65.3321 | 62.4403 | +2.7329 | +7.3918 | no | 54.9010 | +7.5393 |
| (1,2) | 121.1637 | 55.9938 | +5.9144 | +6.2399 | yes | 54.2029 | +1.7909 |
| (10,11) | 78.4111 | 70.0146 | +10.3072 | +17.1039 | yes | 60.7161 | +9.2985 |

## Verdict logic (item 324 verbatim)

- CONFIRMED iff (b10, b11) r=1 delta_vs_worst_single ≥ +5.0 → True
- REFUTED iff (b10, b11) r=1 delta_vs_worst_single ≤ +1.0 AND n_other_destructive ≤ 1 → False
- VERDICT: **CONFIRMED** (CONFIRMED-persistent-destruction)

- worst pair: **[10, 11]** best_mean=70.0146 delta=+10.3072
- best pair:  **[2, 11]** best_mean=58.2418 delta=-1.4656

## Cycle 310 comparison (heterogeneous r vs universal r=1)

| pair | cycle 310 (het r) | cycle 314 (r=1) | delta |
|---|---|---|---|
| (1,2) | 54.2029 | 55.9938 | +1.7909 |
| (1,3) | 51.6544 | 56.0217 | +4.3673 |
| (1,10) | 52.1901 | 53.5124 | +1.3223 |
| (1,11) | 53.5812 | 59.8573 | +6.2761 |
| (2,3) | 52.3641 | 55.6467 | +3.2826 |
| (2,10) | 52.9059 | 54.2429 | +1.3370 |
| (2,11) | 53.8455 | 58.2418 | +4.3963 |
| (3,10) | 52.2278 | 55.4499 | +3.2221 |
| (3,11) | 54.9010 | 62.4403 | +7.5393 |
| (10,11) | 60.7161 | 70.0146 | +9.2985 |

## Single-block r=1 anchors (diagonal)

- b1 r=1: 50.0794 (cycle 292)
- b2 r=1: 49.7539 (cycle 311)
- b3 r=1: 55.0485 (cycle 311)
- b10 r=1: 52.9107 (cycle 311)
- b11 r=1: 59.7074 (cycle 311)
