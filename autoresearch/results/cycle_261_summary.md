# Cycle 261 — W_down sign-frequency geometry (Item 272)

**Verdict: INCONCLUSIVE**

## Setup
- GPT-2 small, calib = 200 sequences × 128 = 25600 tokens, wikitext-2 val (seed 42).
- For each block l ∈ {0..11}: σ_c = E_x[1{z_c>0}], split channels into σ ∈ [0, 0.1] / [0.45, 0.55] / [0.9, 1.0]; SVD of corresponding W_down row subsets; principal angles (top-20 right SVs) between mostly_pos and mostly_neg.
- Null: 100 shuffles preserving (n_pos, n_neg) per block.

## Per-block group sizes and angles

| block | n_neg | n_bal | n_pos | rank95_pos | rank95_neg | pos_top1 | median_angle° | null_mean±std° | inside_2σ |
|------:|-----:|-----:|-----:|----------:|----------:|--------:|------------:|---------------:|:--------:|
| 0 | 2057 | 37 | 145 | 110 | 569 | 0.107 | 76.27 | 69.57±3.95 | Y |
| 1 | 1188 | 489 | 12 | 9 | 487 | 0.299 | 71.87 | 79.00±1.81 | N |
| 2 | 1692 | 186 | 6 | 5 | 536 | 0.552 | 62.30 | 72.89±2.53 | N |
| 3 | 1897 | 80 | 7 | 5 | 551 | 0.613 | 81.08 | 73.64±1.85 | N |
| 4 | 1562 | 64 | 8 | 7 | 536 | 0.536 | 75.28 | 72.74±2.39 | Y |
| 5 | 1665 | 72 | 6 | 5 | 552 | 0.427 | 76.99 | 71.60±2.99 | Y |
| 6 | 1618 | 72 | 7 | 6 | 555 | 0.434 | 73.21 | 74.81±2.39 | Y |
| 7 | 1592 | 54 | 3 | 3 | 557 | 0.767 | 73.59 | 73.95±2.73 | Y |
| 8 | 1587 | 47 | 4 | 3 | 550 | 0.615 | 72.32 | 71.91±3.25 | Y |
| 9 | 1628 | 37 | 2 | 2 | 552 | 0.731 | 21.35 | 69.67±8.53 | N |
| 10 | 1277 | 35 | 5 | 4 | 526 | 0.858 | 75.71 | 77.13±2.02 | Y |
| 11 | 1084 | 49 | 9 | 7 | 504 | 0.685 | 76.96 | 78.51±1.51 | Y |

## Block 2 rank-1 check (cycle 174)
- mostly_pos (σ ≥ 0.9) at block 2: n=6, top-1 SV energy fraction = **0.5517**, rank_95 = 5.
- Threshold for clause (a): top-1 ≥ 0.70 → **FAIL**.

## Verdict logic
- (a) block-2 pos top-1 ≥ 0.70: **False** (value 0.5517)
- (b) median angle blocks 1→11 strictly monotone decreasing AND total drop ≥ 30°: **False** (monotone=False, total_drop=-5.09°)
- (c) blocks 1..11 each outside null mean ± 2σ: **False**
- REFUTED clause (all 12 blocks inside null ±2σ): **False**
- **Verdict: INCONCLUSIVE**
