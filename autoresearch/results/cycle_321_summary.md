# Cycle 321 — Item 330: Zero-shot ReLU² / LeakyReLU² patches on GPT-2 small

**Verdict:** CONFIRMED-distinct-profiles

## Setup
- GPT-2 small, wikitext-2 val, 200 seqs × 128 tokens, seed 42.
- Three (σ, σ_lin) pairs: (ReLU², ReLU), (LeakyReLU²(0.5), ReLU²), (GELU, ReLU).
- Panels run: panel_raw_rank (rank_95(Δy)), panel_centred_alignment (c_fc/Q/K/V top-5 cos sums).
- Panel 5 (κ shift under sigma_lin + b_leak swap) OMITTED to keep wall under budget.
- Trainable params: 0 (zero-shot patches).

## Pair (relu2, relu)

| block | rank_95_raw | rank_99_raw | top1_sv | ‖Δy‖_mean | rank_95_centred | c_fc_top5_cos_sum |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 35 | 222 | 12619.494 | 43.118 | 36 | 1.769 |
| 1 | 1 | 1 | 85437.391 | 66.452 | 1 | 2.220 |
| 2 | 1 | 1 | 1592614.500 | 898.445 | 1 | 2.969 |
| 3 | 1 | 15 | 16553.217 | 23.653 | 1 | 3.512 |
| 4 | 174 | 494 | 3195.904 | 17.989 | 220 | 3.577 |
| 5 | 423 | 640 | 1369.289 | 16.487 | 447 | 3.403 |
| 6 | 443 | 647 | 1367.543 | 19.139 | 467 | 3.265 |
| 7 | 428 | 640 | 1993.629 | 23.792 | 450 | 3.216 |
| 8 | 245 | 530 | 6348.829 | 43.719 | 342 | 3.356 |
| 9 | 213 | 504 | 8838.136 | 59.759 | 309 | 2.780 |
| 10 | 112 | 413 | 17602.129 | 110.874 | 202 | 2.401 |
| 11 | 4 | 113 | 51022.316 | 162.528 | 5 | N/A |

## Pair (lrelu2, relu2)

| block | rank_95_raw | rank_99_raw | top1_sv | ‖Δy‖_mean | rank_95_centred | c_fc_top5_cos_sum |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 155 | 26615.650 | 173.853 | 223 | 2.432 |
| 1 | 1 | 1 | 134413.922 | 824.112 | 5 | 2.870 |
| 2 | 1 | 1 | 298274.031 | 1779.767 | 1 | 2.861 |
| 3 | 1 | 6 | 109999.625 | 674.856 | 72 | 2.790 |
| 4 | 11 | 162 | 38360.512 | 247.563 | 257 | 3.195 |
| 5 | 31 | 224 | 35923.457 | 236.314 | 287 | 3.038 |
| 6 | 95 | 333 | 24128.922 | 167.517 | 316 | 3.181 |
| 7 | 123 | 377 | 22706.771 | 162.696 | 325 | 3.093 |
| 8 | 169 | 426 | 21483.258 | 161.704 | 336 | 3.360 |
| 9 | 195 | 457 | 22052.559 | 177.078 | 333 | 2.817 |
| 10 | 81 | 341 | 43119.836 | 294.061 | 317 | 2.769 |
| 11 | 5 | 97 | 96163.656 | 627.426 | 102 | N/A |

## Pair (gelu, relu)

| block | rank_95_raw | rank_99_raw | top1_sv | ‖Δy‖_mean | rank_95_centred | c_fc_top5_cos_sum |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 322 | 3355.340 | 21.696 | 489 | 2.782 |
| 1 | 132 | 492 | 2625.421 | 17.504 | 554 | 2.705 |
| 2 | 1 | 173 | 5978.789 | 37.876 | 524 | 2.586 |
| 3 | 1 | 100 | 7075.795 | 44.643 | 520 | 2.701 |
| 4 | 1 | 107 | 7207.229 | 45.475 | 526 | 3.318 |
| 5 | 1 | 173 | 6865.655 | 43.502 | 538 | 3.233 |
| 6 | 1 | 278 | 6137.092 | 39.185 | 551 | 3.327 |
| 7 | 8 | 353 | 5816.754 | 37.391 | 565 | 3.356 |
| 8 | 73 | 430 | 5316.456 | 34.707 | 558 | 3.393 |
| 9 | 110 | 461 | 5642.168 | 37.077 | 562 | 2.762 |
| 10 | 1 | 323 | 9698.396 | 61.937 | 567 | 2.493 |
| 11 | 1 | 1 | 24446.512 | 153.284 | 552 | N/A |

## Verdict logic (item 330 verbatim)
- CONFIRMED-distinct-profiles iff (ReLU², ReLU) and (LeakyReLU², ReLU²) produce per-block rank/alignment profiles measurably different from (GELU, ReLU) — at least one measurement differs by ≥30% relative at ≥4 blocks per pair.
- CONFIRMED-identical-profiles iff all 3 pairs give measurements within ±10% relative at every block.
- INCONCLUSIVE otherwise.

- relu2 vs (gelu,relu) — distinct: True
- lrelu2 vs (gelu,relu) — distinct: True
- relu2 vs (gelu,relu) — within ±10% all blocks: False
- lrelu2 vs (gelu,relu) — within ±10% all blocks: False

**Wall:** 44.1 s
