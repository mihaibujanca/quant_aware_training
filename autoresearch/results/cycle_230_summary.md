# Cycle 230 — Item 240: K=8/4/2/1 Adam compression floor sweep

**Verdict: CONFIRMED**

- Criteria: CONFIRMED if K=4 best_mean ≤ 59.28 (within +10% of float); REFUTED if K=8 best_mean ≥ 61.98 (cliff steeper than extrapolation); else INCONCLUSIVE.
- Config: BLOCKS=[1, 4, 7], K_VALUES=[4, 8, 2, 1], LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895

## Extended compression efficiency curve  (K ∈ {512..1})

| K | trainable (3 blocks) | best_ppl_mean | std | %vs float | source |
|---|---|---|---|---|---|
| 512 | 1,181,952 | 46.556 | — | -13.62% | cycle 210 |
| 256 | 592,128 | 47.509 | — | -11.85% | cycle 223 |
| 128 | 297,216 | 48.989 | — | -9.10% | cycle 223 |
| 64 | 149,760 | 50.781 | — | -5.78% | cycle 225 |
| 32 | 76,032 | 53.038 | — | -1.59% | cycle 225 |
| 16 | 39,168 | 55.133 | — | +2.30% | cycle 225 |
| 8 | 20,736 | 56.323 | 0.011 | +4.50% | this cycle |
| 4 | 11,520 | 57.328 | 0.026 | +6.37% | this cycle |
| 2 | 6,912 | 58.289 | 0.035 | +8.15% | this cycle |
| 1 | 4,608 | 59.245 | 0.029 | +9.93% | this cycle |

## Per-K aggregates (this cycle)

### K = 4

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 69.830 | 57.293 | 4000 | 57.293 |
| 137 | 69.830 | 57.354 | 4000 | 57.354 |
| 2024 | 69.830 | 57.336 | 4000 | 57.336 |
| **mean** | 69.830 | **57.328** | 4000 | 57.328 |
| **std**  | 0.000 | 0.026 | 0 | 0.026 |

### K = 8

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 69.223 | 56.331 | 4000 | 56.331 |
| 137 | 69.223 | 56.330 | 4000 | 56.330 |
| 2024 | 69.223 | 56.307 | 4000 | 56.307 |
| **mean** | 69.223 | **56.323** | 4000 | 56.323 |
| **std**  | 0.000 | 0.011 | 0 | 0.011 |

### K = 2

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 70.256 | 58.242 | 4000 | 58.242 |
| 137 | 70.256 | 58.327 | 4000 | 58.327 |
| 2024 | 70.256 | 58.298 | 4000 | 58.298 |
| **mean** | 70.256 | **58.289** | 4000 | 58.289 |
| **std**  | 0.000 | 0.035 | 0 | 0.035 |

### K = 1

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 71.409 | 59.209 | 4000 | 59.209 |
| 137 | 71.409 | 59.281 | 4000 | 59.281 |
| 2024 | 71.409 | 59.246 | 4000 | 59.246 |
| **mean** | 71.409 | **59.245** | 4000 | 59.245 |
| **std**  | 0.000 | 0.029 | 0 | 0.029 |

## Floor diagnostic

- `|best_mean - init_mean| < 1.0` → Adam failed to learn (surface too compressed).
- `best_mean > 2 × float_ppl (107.790)` → catastrophic break.

- K=4: init=69.830 → best=57.328 (Δ=-12.503)
- K=8: init=69.223 → best=56.323 (Δ=-12.900)
- K=2: init=70.256 → best=58.289 (Δ=-11.967)
- K=1: init=71.409 → best=59.245 (Δ=-12.164)

## Comparison to prior cycles

- K=512 (cycle 210): **46.556**
- K=256 (cycle 223): **47.509**
- K=128 (cycle 223): **48.989**
- K=64 (cycle 225): **50.781**
- K=32 (cycle 225): **53.038**
- K=16 (cycle 225): **55.133**
- K=4 (this cycle): **57.328** (Δ vs K=16: +2.195, Δ vs K=512: +10.772)
- K=8 (this cycle): **56.323** (Δ vs K=16: +1.190, Δ vs K=512: +9.767)
- K=2 (this cycle): **58.289** (Δ vs K=16: +3.156, Δ vs K=512: +11.733)
- K=1 (this cycle): **59.245** (Δ vs K=16: +4.112, Δ vs K=512: +12.689)
