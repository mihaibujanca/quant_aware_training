# Cycle 322 — Item 335 Summary

Multi-checkpoint eff_rank census: is the min-eff_rank block stable?

**13L (cycle 318 baseline)** (`13L_640d_322_7k_wd70_batch20k_best`): min eff_rank block = b3 (4.89), frac pos = 0.231, n_blocks = 13
**11L polished** (`wd70_11L_5x_polished`): min eff_rank block = b10 (2.46), frac pos = 0.909, n_blocks = 11
**10L multiskip polished** (`10L_640d_multiskip_22k_wd70_polished`): min eff_rank block = b9 (1.53), frac pos = 0.900, n_blocks = 10

## Per-block eff_rank (side-by-side)

| block | 13L_640d_322_7k_wd70 eff_rank | wd70_11L_5x_polished eff_rank | 10L_640d_multiskip_2 eff_rank |
|---|---|---|---|
| 0 | 187.61 | 187.09 | 212.97 |
| 1 | 160.08 | 223.90 | 229.41 |
| 2 | 86.05 | 191.41 | 195.29 |
| 3 | 4.89 **MIN** | 137.94 | 150.52 |
| 4 | 38.30 | 111.09 | 101.67 |
| 5 | 39.96 | 122.07 | 121.02 |
| 6 | 28.21 | 118.19 | 121.05 |
| 7 | 23.23 | 129.11 | 120.09 |
| 8 | 30.78 | 129.34 | 35.30 |
| 9 | 28.85 | 63.36 | 1.53 **MIN** |
| 10 | 21.55 | 2.46 **MIN** | — |
| 11 | 13.40 | — | — |
| 12 | 6.94 | — | — |

## Verdict

**REFUTED-checkpoint-specific**

Min-block absolute indices: [3, 10, 9] (spread=7)
Min-block fractional positions: ['0.231', '0.909', '0.900'] (spread=0.678)

Verdict logic (item 335):
- CONFIRMED-stable: abs_spread <= 1 AND frac_spread <= 0.1 (b3 is canonical LASER target)
- REFUTED-checkpoint-specific: abs_spread >= 3 OR frac_spread >= 0.3 (re-target per checkpoint)
- INCONCLUSIVE: otherwise

## Cycle 318 baseline comparison (13L, b3 eff_rank ~4.89)

13L checkpoint b3 eff_rank = 4.89 (expected ~4.89 from cycle 318)
