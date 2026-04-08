# Cycle 317 — Item 327: 3-block LASER composition sweep

**Verdict: INCONCLUSIVE**

- Float PPL: 53.8949 (ref 53.895)
- Block order: [1, 2, 3, 10, 11]
- Ranks: {1: 2, 2: 2, 3: 24, 10: 5, 11: 22}
- N_STEPS=4000, LR=0.0001, SEEDS=[42], EVAL_EVERY=100
- Triple sweep wall: 26.1 min

## Per-triple results (sorted by delta_vs_worst_single)

| triple | ranks | trainable | init | best | delta_v_worst | destructive@5 | non-dest@2 | b10_b11? |
|---|---|---|---|---|---|---|---|---|
| (1,3,11) | (2,24,22) | 39,168 | 71.1979 | 54.4815 | +0.0128 | no | yes | no |
| (2,3,11) | (2,24,22) | 39,168 | 82.9718 | 56.2144 | +1.7457 | no | yes | no |
| (1,3,10) | (2,24,5) | 26,112 | 73.0833 | 53.7266 | +2.2607 | no | no | no |
| (2,3,10) | (2,24,5) | 26,112 | 84.5384 | 55.0981 | +3.6322 | no | no | no |
| (1,2,11) | (2,2,22) | 22,272 | 85.3136 | 58.7802 | +4.3115 | no | no | no |
| (3,10,11) | (24,5,22) | 41,472 | 80.9008 | 60.2312 | +5.7625 | yes | no | yes |
| (1,2,10) | (2,2,5) | 9,216 | 87.3491 | 57.5398 | +6.0739 | yes | no | no |
| (1,10,11) | (2,5,22) | 24,576 | 81.7562 | 60.9255 | +6.4568 | yes | no | yes |
| (1,2,3) | (2,2,24) | 23,808 | 105.7510 | 57.0819 | +6.4884 | yes | no | no |
| (2,10,11) | (2,5,22) | 24,576 | 90.1049 | 61.6873 | +7.2186 | yes | no | yes |

## Pair-inheritance (cycle 310 sub-pair deltas per triple)

| triple | sub_pairs | cycle 310 deltas | max_pair_delta | mean_pair_delta |
|---|---|---|---|---|
| (1,3,11) | [[1, 3], [1, 11], [3, 11]] | [+1.06, -0.89, +0.43] | +1.0609 | +0.2019 |
| (2,3,11) | [[2, 3], [2, 11], [3, 11]] | [+1.77, -0.62, +0.43] | +1.7706 | +0.5266 |
| (1,3,10) | [[1, 3], [1, 10], [3, 10]] | [+1.06, +0.72, +0.76] | +1.0609 | +0.8490 |
| (2,3,10) | [[2, 3], [2, 10], [3, 10]] | [+1.77, +1.44, +0.76] | +1.7706 | +1.3242 |
| (1,2,11) | [[1, 2], [1, 11], [2, 11]] | [+4.87, -0.89, -0.62] | +4.8739 | +1.1211 |
| (3,10,11) | [[3, 10], [3, 11], [10, 11]] | [+0.76, +0.43, +6.25] | +6.2474 | +2.4805 |
| (1,2,10) | [[1, 2], [1, 10], [2, 10]] | [+4.87, +0.72, +1.44] | +4.8739 | +2.3460 |
| (1,10,11) | [[1, 10], [1, 11], [10, 11]] | [+0.72, -0.89, +6.25] | +6.2474 | +2.0280 |
| (1,2,3) | [[1, 2], [1, 3], [2, 3]] | [+4.87, +1.06, +1.77] | +4.8739 | +2.5685 |
| (2,10,11) | [[2, 10], [2, 11], [10, 11]] | [+1.44, -0.62, +6.25] | +6.2474 | +2.3547 |

## Verdict logic (item 327 verbatim)

- n_destructive := |{triple : delta_vs_worst_single ≥ +5.0}| = **5** / 10
- (b10,b11)-containing triples (3): all destruct = **True**
- non-(b10,b11) triples (7): all non-destructive@2 = **False**
- CONFIRMED-b10b11-carries-destruction iff both → False
- CONFIRMED-triple-level-pervasive iff n ≥ 8 → False
- REFUTED iff n ≤ 2 → False
- VERDICT: **INCONCLUSIVE**

- worst triple: **[2, 10, 11]** best_mean=61.6873 delta=+7.2186
- best triple: **[1, 3, 11]** best_mean=54.4815 delta=+0.0128

## Comparisons to prior cycles

- cycle 310 pair matrix: (b10,b11) 60.7161 (+6.25 unique destructive pair), (b1,b11) 53.5812 best
- cycle 305 5-stack: 73.006 (+18.54 vs worst-single)
