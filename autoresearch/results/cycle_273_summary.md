# Cycle 273 — Item 283: eff_rank-guided heterogeneous K

**Verdict: REFUTED**

- Criteria: CONFIRMED iff hetero best_mean ≤ 44.5 AND uniform_k22 best_mean ≥ 60.0; REFUTED iff hetero ≥ 47.0; else INCONCLUSIVE
- BLOCKS = [1, 4, 7, 11]
- hetero K_PER_BLOCK = {1: 512, 4: 512, 7: 512, 11: 22}; trainable = 1,199,616
- uniform_k22 K_PER_BLOCK = {1: 22, 4: 22, 7: 22, 11: 22}; trainable = 70,656
- LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Float PPL: 53.895

## Per-config aggregates

| cfg | trainable | init_mean | best_mean ± std | best_step | final@4000 |
|---|---|---|---|---|---|
| hetero | 1,199,616 | 72.198 | **52.637 ± 0.059** | 2800 ± 163 | 52.995 |
| uniform_k22 | 70,656 | 82.787 | **60.686 ± 0.048** | 4000 ± 0 | 60.686 |

## Per-seed details

### hetero (K_PER_BLOCK = {'1': 512, '4': 512, '7': 512, '11': 22})

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 72.198 | 52.665 | 2600 | 53.059 |
| 137 | 72.198 | 52.691 | 2800 | 53.092 |
| 2024 | 72.198 | 52.554 | 3000 | 52.835 |

### uniform_k22 (K_PER_BLOCK = {'1': 22, '4': 22, '7': 22, '11': 22})

| seed | init | best | best_step | final@4000 |
|---|---|---|---|---|
| 42 | 82.787 | 60.752 | 4000 | 60.752 |
| 137 | 82.787 | 60.670 | 4000 | 60.670 |
| 2024 | 82.787 | 60.637 | 4000 | 60.637 |

## Per-block residual MSE (lstsq init, seed 42)

### hetero

| block | K | residual MSE |
|---|---|---|
| 1 | 512 | 8.4321e-02 |
| 4 | 512 | 1.6163e-01 |
| 7 | 512 | 4.1485e-01 |
| 11 | 22 | 5.2002e+00 |

### uniform_k22

| block | K | residual MSE |
|---|---|---|
| 1 | 22 | 1.6103e-01 |
| 4 | 22 | 2.6408e-01 |
| 7 | 22 | 6.9959e-01 |
| 11 | 22 | 5.2002e+00 |

## Trajectory snippets

### hetero
- seed 42: init=72.20, step1100=54.42, step2100=52.91, best=step2600@52.66, step3100=52.77, step4000=53.06
- seed 137: init=72.20, step1100=54.52, step2100=53.15, best=step2800@52.69, step3100=52.88, step4000=53.09
- seed 2024: init=72.20, step1100=54.63, step2100=52.95, best=step3000@52.55, step3100=52.72, step4000=52.83

### uniform_k22
- seed 42: init=82.79, step1100=66.09, step2100=63.03, best=step4000@60.75, step3100=61.54, step4000=60.75
- seed 137: init=82.79, step1100=66.12, step2100=63.09, best=step4000@60.67, step3100=61.60, step4000=60.67
- seed 2024: init=82.79, step1100=66.27, step2100=63.04, best=step4000@60.64, step3100=61.51, step4000=60.64

## Comparison

- Cycle 266 K=512 2-block [1,4]:    **43.852**
- Cycle 270 block 11 rank-22 retrain: 54.469
- Cycle 247 hetero K (6 blocks):    64.450
- Cycle 255 K=512 4-block min:      49.609
- Cycle 268 K=64 4-block min:       55.953
- Float ref:                        53.895
- **hetero best_mean: 52.637**
- **uniform_k22 best_mean: 60.686**
- Δ hetero vs cycle 266 [1,4]: **+8.785** PPL
- Δ hetero vs cycle 255 K=512 4-block: **+3.028** PPL
- Δ hetero vs float (run): **-1.258** PPL
