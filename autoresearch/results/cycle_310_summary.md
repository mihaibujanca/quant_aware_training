# Cycle 310 — Item 320: Pairwise LASER composition matrix

**Verdict: REFUTED**  (n_destructive = 1 / 10; CONFIRMED iff ≥6, REFUTED iff ≤2)

- Float PPL: 53.8949 (ref 53.895)
- Block order: [1, 2, 3, 10, 11]
- Ranks: {1: 2, 2: 2, 3: 24, 10: 5, 11: 22}
- N_STEPS=2000, LR=0.0001, SEEDS=[42], EVAL_EVERY=100
- Pair sweep wall: 12.9 min

## 5×5 pair PPL matrix (diagonal = single-block anchor)

| block | 1 | 2 | 3 | 10 | 11 |
|---|---|---|---|---|---|
| **1** | *49.0324* | 54.2029 | 51.6544 | 52.1901 | 53.5812 |
| **2** | 54.2029 | *49.3290* | 52.3641 | 52.9059 | 53.8455 |
| **3** | 51.6544 | 52.3641 | *50.5935* | 52.2278 | 54.9010 |
| **10** | 52.1901 | 52.9059 | 52.2278 | *51.4659* | 60.7161 |
| **11** | 53.5812 | 53.8455 | 54.9010 | 60.7161 | *54.4687* |

## Per-pair detail (sorted by delta_vs_worst_single)

| pair | ranks | init | best | delta_vs_worst_single | delta_vs_best_single | destructive |
|---|---|---|---|---|---|---|
| (1,11) | (2,22) | 63.4629 | 53.5812 | -0.8875 | +4.5488 | no |
| (2,11) | (2,22) | 70.8544 | 53.8455 | -0.6232 | +4.5165 | no |
| (3,11) | (24,22) | 62.9762 | 54.9010 | +0.4323 | +4.3075 | no |
| (1,10) | (2,5) | 64.4650 | 52.1901 | +0.7242 | +3.1577 | no |
| (3,10) | (24,5) | 63.9054 | 52.2278 | +0.7619 | +1.6343 | no |
| (1,3) | (2,24) | 64.3845 | 51.6544 | +1.0609 | +2.6220 | no |
| (2,10) | (2,5) | 71.7644 | 52.9059 | +1.4400 | +3.5769 | no |
| (2,3) | (2,24) | 75.0811 | 52.3641 | +1.7706 | +3.0351 | no |
| (1,2) | (2,2) | 76.8020 | 54.2029 | +4.8739 | +5.1705 | no |
| (10,11) | (5,22) | 76.6535 | 60.7161 | +6.2474 | +9.2502 | yes |

## Verdict logic (item 320 verbatim)

- n_destructive := |{pair : delta_vs_worst_single ≥ +5.0}| = **1** / 10
- CONFIRMED iff n ≥ 6  →  False
- REFUTED   iff n ≤ 2  →  True
- VERDICT: **REFUTED**

- worst pair: **[10, 11]** best_mean=60.7161 delta=+6.2474
- best pair:  **[1, 11]** best_mean=53.5812 delta=-0.8875

## Comparisons to prior cycles

- cycle 284 (b1,b11) 4000-step: 52.8803
- cycle 291 (b1,b10) 4000-step: 51.8008
- cycle 305 5-stack:           73.006

Note: cycle 310 uses N_STEPS=2000 (item 320 budget), cycles 284/291 used 4000;
cycle 310 per-pair best_ppl is expected to be slightly worse than the
matched-anchor 4000-step fits of 284/291 at the same pair. Sanity target:
cycle 310 (1,11) best_mean close to cycle 284 52.8803 ± a few PPL (budget).
