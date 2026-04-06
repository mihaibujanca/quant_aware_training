# Cycle 225 — Item 235: K=64/32/16 Adam compression cliff sweep

**Verdict: CONFIRMED**

- Criteria: CONFIRMED if any best_mean ≤ 53.0 (any K still beats float); REFUTED if all best_mean ≥ 60.0 (real channel-capacity floor); else INCONCLUSIVE.
- Config: BLOCKS=[1, 4, 7], K_VALUES=[64, 32, 16], LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895

## Extended compression efficiency curve

| K | trainable (3 blocks) | best_ppl_mean | std | %vs float | source |
|---|---|---|---|---|---|
| 512 | 1,181,952 | 46.556 | — | -13.62% | cycle 210 |
| 256 | 592,128 | 47.509 | — | -11.85% | cycle 223 |
| 128 | 297,216 | 48.989 | — | -9.10% | cycle 223 |
| 64 | 149,760 | 50.781 | 0.021 | -5.78% | this cycle |
| 32 | 76,032 | 53.038 | 0.009 | -1.59% | this cycle |
| 16 | 39,168 | 55.133 | 0.010 | +2.30% | this cycle |

## Per-K aggregates (this cycle)

### K = 64

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 66.036 | 50.763 | 3900 | 50.765 |
| 137 | 66.036 | 50.770 | 4000 | 50.770 |
| 2024 | 66.036 | 50.810 | 4000 | 50.810 |
| **mean** | 66.036 | **50.781** | 3967 | 50.781 |
| **std**  | 0.000 | 0.021 | 47 | 0.020 |

### K = 32

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 67.033 | 53.032 | 4000 | 53.032 |
| 137 | 67.033 | 53.052 | 4000 | 53.052 |
| 2024 | 67.033 | 53.032 | 4000 | 53.032 |
| **mean** | 67.033 | **53.038** | 4000 | 53.038 |
| **std**  | 0.000 | 0.009 | 0 | 0.009 |

### K = 16

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 68.978 | 55.140 | 4000 | 55.140 |
| 137 | 68.978 | 55.119 | 4000 | 55.119 |
| 2024 | 68.978 | 55.139 | 4000 | 55.139 |
| **mean** | 68.978 | **55.133** | 4000 | 55.133 |
| **std**  | 0.000 | 0.010 | 0 | 0.010 |

## Comparison to cycle 223

- K=512 (cycle 210): **46.556**
- K=256 (cycle 223): **47.509** (Δ vs K=512: +0.953)
- K=128 (cycle 223): **48.989** (Δ vs K=512: +2.433)
- K=64 (this cycle): **50.781** (Δ vs K=512: +4.225, Δ vs K=128: +1.792)
- K=32 (this cycle): **53.038** (Δ vs K=512: +6.482, Δ vs K=128: +4.049)
- K=16 (this cycle): **55.133** (Δ vs K=512: +8.577, Δ vs K=128: +6.144)
