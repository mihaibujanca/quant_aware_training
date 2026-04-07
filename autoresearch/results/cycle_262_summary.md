# Cycle 262 — Item 268: K=1 + c_fc bias trainable (3 blocks)

**Verdict: INCONCLUSIVE**

- Criteria: CONFIRMED if best_mean ≤ 58.5; REFUTED if ≥ 59.5; else INCONCLUSIVE.
- Trainable: 3 × (W_out 768 + b_out 768 + b_fc 1) = 4611 params.
- Config: BLOCKS=[1, 4, 7], K=1, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024].
- Float PPL: 53.895

## Aggregate

| metric | mean | std |
|---|---|---|
| init | 71.409 | 0.000 |
| **best** | **59.131** | 0.027 |
| best_step | 4000 | 0 |
| final@4000 | 59.131 | 0.027 |

## Per-seed

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 71.409 | 59.098 | 4000 | 59.098 |
| 137 | 71.409 | 59.165 | 4000 | 59.165 |
| 2024 | 71.409 | 59.130 | 4000 | 59.130 |

## Comparison to baselines

- **Cycle 230 K=1 3-block W_out-only**: 59.245 (baseline).
- **This cycle (K=1 3-block, +b_fc)**: 59.131.
- **Delta vs cycle 230**: -0.114 PPL.
- Cycle 213 K=512 3-block W_out+c_fc bias: noop.
- Cycle 241 K=512 6-block W_out+c_fc bias: noop (59.533 vs 59.515).
- Cycle 240 K=1 6-block: 96.886 (catastrophic).
- Float PPL: 53.895.

## b_fc trajectories

| seed | b_fc[block 1] | b_fc[block 4] | b_fc[block 7] |
|---|---|---|---|
| 42 | +0.3817 | +0.3530 | +1.1657 |
| 137 | +0.4199 | +0.3584 | +1.1664 |
| 2024 | +0.4260 | +0.3560 | +1.1655 |
