# Cycle 026 Summary — Rank vs Geometry vs Distillation

**Baseline PPL (uncompressed)**: 122.35
**Verdict**: CONFIRMED
**Detail**: Geometric beats SVD in 11/12 (92%) pairs.

## Per-cell table

| Block | Ratio | Method | PPL | ΔPPL% | BlockMSE |
|-------|-------|--------|-----|-------|----------|
| 2 | 4× | svd_composed | 4841.50 | +3857.1% | 799.7686 |
| 2 | 4× | svd_norm | 125.99 | +3.0% | 0.0574 |
| 2 | 4× | geometric | 121.46 | -0.7% | 0.0491 |
| 2 | 4× | distillation | 122.13 | -0.2% | 55.3673 |
| 2 | 8× | svd_composed | 4429.24 | +3520.1% | 799.5898 |
| 2 | 8× | svd_norm | 126.15 | +3.1% | 0.1205 |
| 2 | 8× | geometric | 123.84 | +1.2% | 0.1022 |
| 2 | 8× | distillation | 122.51 | +0.1% | 55.3625 |
| 2 | 16× | svd_composed | 4458.75 | +3544.3% | 798.9555 |
| 2 | 16× | svd_norm | 126.86 | +3.7% | 0.1671 |
| 2 | 16× | geometric | 123.37 | +0.8% | 0.1480 |
| 2 | 16× | distillation | 122.78 | +0.3% | 55.3659 |
| 5 | 4× | svd_composed | 3813.55 | +3016.9% | 165.3123 |
| 5 | 4× | svd_norm | 130.94 | +7.0% | 0.1242 |
| 5 | 4× | geometric | 123.31 | +0.8% | 0.1142 |
| 5 | 4× | distillation | 116.71 | -4.6% | 0.5610 |
| 5 | 8× | svd_composed | 3807.37 | +3011.9% | 165.0876 |
| 5 | 8× | svd_norm | 127.28 | +4.0% | 0.2328 |
| 5 | 8× | geometric | 122.07 | -0.2% | 0.2174 |
| 5 | 8× | distillation | 124.69 | +1.9% | 0.5602 |
| 5 | 16× | svd_composed | 3336.08 | +2626.7% | 164.3035 |
| 5 | 16× | svd_norm | 125.07 | +2.2% | 0.3274 |
| 5 | 16× | geometric | 121.99 | -0.3% | 0.3033 |
| 5 | 16× | distillation | 122.91 | +0.5% | 0.5607 |
| 8 | 4× | svd_composed | 13875.01 | +11240.4% | 71.6114 |
| 8 | 4× | svd_norm | 127.28 | +4.0% | 0.2671 |
| 8 | 4× | geometric | 124.03 | +1.4% | 0.2238 |
| 8 | 4× | distillation | 125.11 | +2.3% | 1.2704 |
| 8 | 8× | svd_composed | 13330.68 | +10795.5% | 71.2934 |
| 8 | 8× | svd_norm | 125.85 | +2.9% | 0.5104 |
| 8 | 8× | geometric | 123.48 | +0.9% | 0.4328 |
| 8 | 8× | distillation | 126.28 | +3.2% | 1.2713 |
| 8 | 16× | svd_composed | 13418.37 | +10867.2% | 70.1132 |
| 8 | 16× | svd_norm | 125.40 | +2.5% | 0.7102 |
| 8 | 16× | geometric | 124.79 | +2.0% | 0.5970 |
| 8 | 16× | distillation | 125.86 | +2.9% | 1.2718 |
| 11 | 4× | svd_composed | 320107285.65 | +261632791.3% | 814.0464 |
| 11 | 4× | svd_norm | 131.17 | +7.2% | 0.5029 |
| 11 | 4× | geometric | 128.06 | +4.7% | 0.3700 |
| 11 | 4× | distillation | 132.91 | +8.6% | 17.2657 |
| 11 | 8× | svd_composed | 359739093.87 | +294025008.0% | 813.6245 |
| 11 | 8× | svd_norm | 128.43 | +5.0% | 1.1203 |
| 11 | 8× | geometric | 128.27 | +4.8% | 0.8918 |
| 11 | 8× | distillation | 134.05 | +9.6% | 17.2690 |
| 11 | 16× | svd_composed | 430240399.53 | +351647752.8% | 811.9683 |
| 11 | 16× | svd_norm | 128.98 | +5.4% | 1.8058 |
| 11 | 16× | geometric | 129.45 | +5.8% | 1.4676 |
| 11 | 16× | distillation | 134.68 | +10.1% | 17.2648 |

## Win/Loss: geometric vs SVD

- Block 2, 4×: geo=121.46 svd_norm=125.99 → **geometric**
- Block 2, 8×: geo=123.84 svd_norm=126.15 → **geometric**
- Block 2, 16×: geo=123.37 svd_norm=126.86 → **geometric**
- Block 5, 4×: geo=123.31 svd_norm=130.94 → **geometric**
- Block 5, 8×: geo=122.07 svd_norm=127.28 → **geometric**
- Block 5, 16×: geo=121.99 svd_norm=125.07 → **geometric**
- Block 8, 4×: geo=124.03 svd_norm=127.28 → **geometric**
- Block 8, 8×: geo=123.48 svd_norm=125.85 → **geometric**
- Block 8, 16×: geo=124.79 svd_norm=125.40 → **geometric**
- Block 11, 4×: geo=128.06 svd_norm=131.17 → **geometric**
- Block 11, 8×: geo=128.27 svd_norm=128.43 → **geometric**
- Block 11, 16×: geo=129.45 svd_norm=128.98 → **svd_norm**
