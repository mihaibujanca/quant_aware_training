# Cycle 306 — Item 316: Block 8 deep dive (curvature residual)

**Verdict: REFUTED**

(a) b8 W_proj top2/top5 ratio = 1.4875; concentrator median = 1.5142; threshold (≤ 0.5×median) = 0.7571; cond_a (b8 flatter) = False. (b) b8 Dy eff_rank = 1.20; concentrator median = 1.08; threshold (≥ 1.5×median) = 1.62; cond_b (b8 broader) = False. CONFIRMED iff (a) AND (b); REFUTED iff NOT (a) AND NOT (b); else INCONCLUSIVE.

## Geometric object
For GPT-2 small block l, z = LN2(x) @ W_fc + b_fc in R^3072, delta = GELU(z) - ReLU(z) in R^3072, Dy = delta @ W_proj in R^768 (W_proj HF Conv1D (3072, 768)). Two SVDs: W_proj = U_w diag(S_w) Vh_w; Dy = U_y diag(S_y) Vh_y.

## Setup
- Calib: 200 wikitext-2 val seqs × 128 tokens (seed 42); N_TOKENS = 25600.
- Float PPL on calib slice = 45.0637 (ref 45.0637).
- Primary blocks: [7, 8, 9].  Concentrator reference: [1, 2, 10, 11].
- Trainable params: 0.

## W_proj spectrum & Δy spectrum per block

| block | s_w[0] | s_w[1] | s_w[4] | top2/top5 | s_y[0] | Δy eff_rank | Δy k_95 | top1 E | top5 E | E_kept_top_2 (W_proj basis) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.706 | 23.104 | 13.739 | 1.682 | 2.625e+03 | 1.31 | 132 | 0.8736 | 0.8957 | 0.7463 |
| 2 | 38.255 | 15.165 | 11.579 | 1.310 | 5.979e+03 | 1.06 | 1 | 0.9692 | 0.9731 | 0.9190 |
| 7 | 30.227 | 16.795 | 14.831 | 1.132 | 5.817e+03 | 1.12 | 8 | 0.9435 | 0.9484 | 0.6622 |
| 8 | 31.076 | 24.495 | 16.467 | 1.488 | 5.316e+03 | 1.20 | 73 | 0.9141 | 0.9226 | 0.4722 |
| 9 | 32.782 | 21.967 | 18.312 | 1.200 | 5.642e+03 | 1.23 | 110 | 0.9017 | 0.9101 | 0.5356 |
| 10 | 41.585 | 28.194 | 19.657 | 1.434 | 9.698e+03 | 1.10 | 1 | 0.9540 | 0.9584 | 0.8374 |
| 11 | 60.993 | 34.594 | 21.700 | 1.594 | 2.445e+04 | 1.02 | 1 | 0.9910 | 0.9920 | 0.9803 |

## Per-token Δy norms

| block | mean | std | q05 | median | q95 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.504 | 1.345 | 15.272 | 17.539 | 19.677 | 23.390 |
| 2 | 37.876 | 2.481 | 33.587 | 38.088 | 41.630 | 45.517 |
| 7 | 37.391 | 1.659 | 34.611 | 37.433 | 40.081 | 43.780 |
| 8 | 34.707 | 1.800 | 31.442 | 34.796 | 37.503 | 40.927 |
| 9 | 37.077 | 2.096 | 33.334 | 37.241 | 40.241 | 43.641 |
| 10 | 61.937 | 3.918 | 54.961 | 62.286 | 67.845 | 74.681 |
| 11 | 153.284 | 7.816 | 139.775 | 153.908 | 164.527 | 178.435 |

## Top-5 right-SV cosine alignment Δy ↔ W_proj

Principal-angle cosines of (Vh_y[:5], Vh_w[:5]); both are orthonormal 5-frames in R^768. cos_top5_sum max = 5.0 (perfect overlap), 0 = orthogonal.

| block | cos_1 | cos_2 | cos_3 | cos_4 | cos_5 | sum top-5 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9541 | 0.8595 | 0.6721 | 0.1436 | 0.0076 | 2.6370 |
| 2 | 0.9797 | 0.7068 | 0.2432 | 0.1697 | 0.0152 | 2.1145 |
| 7 | 0.9158 | 0.7768 | 0.5642 | 0.2740 | 0.0824 | 2.6133 |
| 8 | 0.8505 | 0.8011 | 0.5154 | 0.4357 | 0.0481 | 2.6508 |
| 9 | 0.8861 | 0.6802 | 0.3416 | 0.1428 | 0.0640 | 2.1147 |
| 10 | 0.9502 | 0.8602 | 0.4643 | 0.1992 | 0.0323 | 2.5062 |
| 11 | 0.9954 | 0.9290 | 0.8884 | 0.1806 | 0.0862 | 3.0796 |

## Item 316 verdict logic (verbatim)
- **CONFIRMED** iff (a) b8 W_proj top-2/top-5 SV ratio ≤ 0.5 × median concentrator ratio AND (b) b8 Δy eff_rank ≥ 1.5 × median concentrator Δy eff_rank.
- **REFUTED** iff neither (a) nor (b) holds.
- **INCONCLUSIVE** otherwise.

- Concentrator W_proj top2/top5 ratios = [1.6815, 1.3097, 1.4343, 1.5942], median = 1.5142, 0.5× = 0.7571.
- Concentrator Δy eff_rank = [1.31, 1.06, 1.1, 1.02], median = 1.08, 1.5× = 1.62.
- b8: top2/top5 = 1.4875 (≤ 0.7571? False); Δy eff_rank = 1.20 (≥ 1.62? False).

**Result: REFUTED**.

## Artifacts
- experiments/block8_deep_dive_cycle306.py
- experiments/curvature_residual_geometric.py
- autoresearch/results/cycle_306_block8_deep_dive.json
- autoresearch/results/cycle_306_panels.png
- autoresearch/results/cycle_306_summary.md
