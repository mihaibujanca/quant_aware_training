# Cycle 227 — Item 237: Δ̃_l specificity control vs non-adjacent c_fc(j)

**Verdict:** INCONCLUSIVE

## Headline
- median_specificity (max@l+1 − mean_nonadj, top-5 cos_sum): **+0.791** (CONFIRMED ≥ 1.0, REFUTED < 0.3)
- rows where all non-adjacent (l, j) sit within mean+3σ of random: **0 / 10** (CONFIRMED ≥ 7)
- non-adjacent entries above 3σ random: **55 / 55**
- sanity check (cycle 227 adjacent top5 mean = 2.9687 vs cycle 218 2.9687): OK

## Heatmap — top-5 principal cosine sum (rows l, cols j)

| l\j | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 2.78 | 2.27 | 1.98 | 1.67 | 1.76 | 1.75 | 1.77 | 1.79 | 1.64 | 1.52 | 1.68 |
| 1 | · | 2.70 | 2.21 | 1.92 | 1.95 | 1.83 | 1.88 | 1.88 | 1.80 | 1.69 | 1.63 |
| 2 | · | · | 2.59 | 2.06 | 2.12 | 1.99 | 1.99 | 1.94 | 1.87 | 1.82 | 1.73 |
| 3 | · | · | · | 2.70 | 2.71 | 2.64 | 2.50 | 2.40 | 2.19 | 1.91 | 1.80 |
| 4 | · | · | · | · | 3.32 | 3.02 | 2.75 | 2.50 | 2.25 | 1.88 | 1.69 |
| 5 | · | · | · | · | · | 3.23 | 3.05 | 2.82 | 2.56 | 2.02 | 1.72 |
| 6 | · | · | · | · | · | · | 3.33 | 3.19 | 2.99 | 2.45 | 2.13 |
| 7 | · | · | · | · | · | · | · | 3.36 | 3.07 | 2.48 | 2.16 |
| 8 | · | · | · | · | · | · | · | · | 3.39 | 2.73 | 2.30 |
| 9 | · | · | · | · | · | · | · | · | · | 2.76 | 2.32 |
| 10 | · | · | · | · | · | · | · | · | · | · | 2.49 |

## Heatmap — top-20 principal cosine sum

| l\j | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| 0 | 4.85 | 3.72 | 3.53 | 3.16 | 3.39 | 3.31 | 3.53 | 3.65 | 3.50 | 3.21 | 3.40 |
| 1 | · | 4.42 | 4.05 | 3.53 | 3.77 | 3.63 | 3.88 | 3.89 | 3.65 | 3.35 | 3.31 |
| 2 | · | · | 4.86 | 3.95 | 4.16 | 4.04 | 4.01 | 4.07 | 3.97 | 3.91 | 3.64 |
| 3 | · | · | · | 5.23 | 5.54 | 5.18 | 5.06 | 4.90 | 4.51 | 3.85 | 3.61 |
| 4 | · | · | · | · | 6.90 | 5.99 | 5.55 | 5.12 | 4.65 | 3.64 | 3.49 |
| 5 | · | · | · | · | · | 6.58 | 6.11 | 5.81 | 5.19 | 4.10 | 3.54 |
| 6 | · | · | · | · | · | · | 6.71 | 6.40 | 5.92 | 4.72 | 4.13 |
| 7 | · | · | · | · | · | · | · | 7.07 | 6.25 | 5.06 | 4.31 |
| 8 | · | · | · | · | · | · | · | · | 6.39 | 5.10 | 4.40 |
| 9 | · | · | · | · | · | · | · | · | · | 5.18 | 4.50 |
| 10 | · | · | · | · | · | · | · | · | · | · | 4.56 |

## Per-row specificity (top-5 cos_sum)

| l | max @ l+1 | mean over j>l+1 | specificity | all non-adj quiet? |
|---:|---------:|----------------:|------------:|:-----------------:|
| 0 | 2.782 | 1.783 | +0.999 | · |
| 1 | 2.705 | 1.864 | +0.841 | · |
| 2 | 2.586 | 1.939 | +0.647 | · |
| 3 | 2.701 | 2.306 | +0.396 | · |
| 4 | 3.318 | 2.349 | +0.969 | · |
| 5 | 3.233 | 2.435 | +0.798 | · |
| 6 | 3.327 | 2.690 | +0.637 | · |
| 7 | 3.356 | 2.572 | +0.784 | · |
| 8 | 3.393 | 2.513 | +0.880 | · |
| 9 | 2.762 | 2.315 | +0.446 | · |
| 10 | 2.493 | — | — | — |

Median specificity (rows l=0..9): **+0.791**

## Random baseline — per j (10 seeds Gaussian QR-orthonormal 20×768)

| j | top5 mean | top5 std | top5 mean+3σ | top20 mean | top20 std |
|---:|---------:|--------:|------------:|----------:|---------:|
| 1 | 1.2715 | 0.0539 | 1.4331 | 2.7116 | 0.1502 |
| 2 | 1.2631 | 0.0529 | 1.4219 | 2.7234 | 0.1030 |
| 3 | 1.2561 | 0.0578 | 1.4295 | 2.7346 | 0.1248 |
| 4 | 1.2618 | 0.0348 | 1.3662 | 2.7216 | 0.0690 |
| 5 | 1.2629 | 0.0335 | 1.3634 | 2.7235 | 0.1035 |
| 6 | 1.2592 | 0.0466 | 1.3989 | 2.7246 | 0.1034 |
| 7 | 1.2591 | 0.0469 | 1.3998 | 2.7367 | 0.1049 |
| 8 | 1.2629 | 0.0452 | 1.3984 | 2.7327 | 0.0948 |
| 9 | 1.2529 | 0.0291 | 1.3402 | 2.6651 | 0.0708 |
| 10 | 1.2119 | 0.0439 | 1.3438 | 2.6598 | 0.0594 |
| 11 | 1.2517 | 0.0335 | 1.3521 | 2.7152 | 0.0706 |

## Plain-language summary
Mixed evidence. Median specificity = 0.79 (in the 0.3–1.0 inconclusive band), and 0/10 rows have all non-adjacent entries within 3σ of random. The peak at j=l+1 IS larger than the average non-adjacent value, but the gap is small enough that the 'generic reading basis' interpretation cannot be ruled out from this measurement alone. A direct functional test (e.g. swap c_fc(l+1) vs c_fc(j>l+1) and measure block l write→read effect) is needed to disambiguate.

## Tie back to cycle 218
Cycle 218 reported 11/11 cross-block pairs at +17–33σ above random for the c_fc(l+1) reader. This cycle reproduces the adjacent-pair signal (cycle 227 mean cos_sum_top5 over (l, l+1) = 2.9687, cycle 218 = 2.9687, Δ = -0.0000) and asks the missing question: does that alignment also appear at j > l+1? The INCONCLUSIVE verdict above answers it.
