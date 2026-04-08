# Cycle 318 — Item 333 Summary

Output-covariance eff_rank census on parameter-golf 13L_640d ReLU² checkpoint.

**Checkpoint**: `/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz`
**n_tokens**: 25600 (1024-token seqs)
**Activation**: ReLU² (no bias on `mlp.fc` / `mlp.proj`)

## Per-block profile (pre-`mlp_scale` MLP outputs)

| block | dim | eff_rank | rank_95 | rank_99 | top1 / λ_sum |
|---|---|---|---|---|---|
| 0 | 640 | 187.61 | 378 | 542 | 0.1196 |
| 1 | 640 | 160.08 | 367 | 534 | 0.1836 |
| 2 | 640 | 86.05 | 314 | 499 | 0.2833 |
| 3 | 640 | 4.89 | 94 | 309 | 0.7822 |
| 4 | 640 | 38.30 | 198 | 396 | 0.3730 |
| 5 | 640 | 39.96 | 201 | 401 | 0.3685 |
| 6 | 640 | 28.21 | 171 | 365 | 0.4319 |
| 7 | 640 | 23.23 | 144 | 323 | 0.4477 |
| 8 | 640 | 30.78 | 164 | 346 | 0.3938 |
| 9 | 640 | 28.85 | 168 | 358 | 0.4137 |
| 10 | 640 | 21.55 | 150 | 326 | 0.4824 |
| 11 | 640 | 13.40 | 114 | 280 | 0.5691 |
| 12 | 640 | 6.94 | 38 | 92 | 0.6510 |

**Min eff_rank**: 4.89 (block 3)
**Max eff_rank**: 187.61 (block 0)
**# blocks ≤ 20**: 3 / 13, **# blocks ≥ 100**: 2 / 13

## Comparison to qat cycle 278 (GPT-2 small, 12 blocks)

GPT-2 small showed a U-shape: extreme collapse at b1/b2 (eff_rank ~1-2), recovery to ~150-200 mid-stack, narrowing again at b11 (~22).

| block | GPT-2 (cycle 278) | parameter-golf 13L_640d |
|---|---|---|
| 0 | 71.93 | 187.61 |
| 1 | 1.94 | 160.08 |
| 2 | 1.05 | 86.05 |
| 3 | 24.29 | 4.89 |
| 4 | 49.04 | 38.30 |
| 5 | 179.19 | 39.96 |
| 6 | 290.97 | 28.21 |
| 7 | 297.79 | 23.23 |
| 8 | 291.41 | 30.78 |
| 9 | 195.50 | 28.85 |
| 10 | 16.70 | 21.55 |
| 11 | 22.49 | 13.40 |
| 12 | — | 6.94 |

## Verdict

**CONFIRMED-different-shape**

Verdict logic (item 333):
- CONFIRMED-collapse-generalizes: any block eff_rank ≤ 20
- CONFIRMED-no-slack: all blocks eff_rank ≥ 100
- CONFIRMED-different-shape: some ≤ 20, some ≥ 100
- INCONCLUSIVE: all in (20, 100)

**Trainable params**: 0 (forward-only measurement, no optimizer step).
