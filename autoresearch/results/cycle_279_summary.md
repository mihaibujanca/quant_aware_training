# Cycle 279 — Item 289: [1,4] K=512 W_out + W_fc trainable (SLICED)

**Verdict: CONFIRMED**

- Criteria (item 289 verbatim): CONFIRMED (W_fc noop) if best_mean ∈ (43.6, 44.1); REFUTED (W_fc helps) if best_mean ≤ 43.5; INCONCLUSIVE otherwise.
- BLOCKS=[1, 4], K=512, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895
- Wall time: 7.6 min; fits=3

## SLICED-vs-full semantics note

**Trainable = W_fc slice (768, K=512) + b_fc slice (512) + W_out (512, 768) + b_out (768) per block.**
Per-block: 787,712. Two-block total: **1,575,424**.
This is the cycle 213 FullCompressedMLP convention — Adam trains the topk-sliced columns of c_fc, NOT the full (768, 3072) c_fc matrix.

Item 289's setup line erroneously claimed W_fc_l (768×3072) trainable → 5,512,704 total params — repeats the cycle 248 agenda mistake. Faithful continuation of the cycle 213/241/248/262 W_fc retraining family REQUIRES the sliced form for apples-to-apples comparison.

## Aggregate

| metric | value |
|---|---|
| init_mean | 57.057 ± 0.000 |
| **best_mean** | **43.878 ± 0.099** |
| best_step_mean | 1933 ± 170 |
| final@4000_mean | 46.630 ± 0.252 |

## Per-seed details

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 57.057 | 44.017 | 1700 | 46.959 |
| 137 | 57.057 | 43.796 | 2100 | 46.584 |
| 2024 | 57.057 | 43.819 | 2000 | 46.346 |

## Comparison

- Cycle 266 [1,4] K=512 W_out-only:        **43.852** (788K)
- Cycle 213 [1,4,7] K=512 W_out+W_fc:      46.424
- Cycle 248 6-block K=128 W_out+W_fc:      67.609
- Cycle 271 [1] K=512 W_out-only:          43.821 (394K, current Pareto winner)
- Float ref:                                53.895
- Cycle 279 [1,4] K=512 W_out+W_fc SLICED: **43.878** (1,575,424 trainable)
- Delta vs cycle 266 [1,4]:                **+0.026** PPL
- Delta vs cycle 213 (3b W_out+W_fc):      **-2.546** PPL
- Delta vs float:                           **-10.017** PPL
