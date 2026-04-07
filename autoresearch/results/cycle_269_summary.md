# Cycle 269 — Item 279: K=1 6-block extended budget (16000 steps)

**Verdict: CONFIRMED**

- Criteria: CONFIRMED (structural) if best_mean ≥ 90.0; REFUTED (budget-limited) if best_mean ≤ 80.0; INCONCLUSIVE otherwise.
- Config: BLOCKS=[1, 2, 4, 5, 7, 8], K=1, LR=0.0001, N_STEPS=16000, EVAL_EVERY=400, SEEDS=[42, 137, 2024]
- Trainable params: 9,216 (6 × (768 + 768))
- Float PPL: 53.895

## Per-seed results

| seed | init | best | best_step | final@16000 |
|---|---|---|---|---|
| 42 | 203.246 | 91.657 | 16000 | 91.657 |
| 137 | 203.246 | 91.613 | 16000 | 91.613 |
| 2024 | 203.246 | 91.568 | 16000 | 91.568 |
| **mean** | 203.246 | **91.613** | 16000 | 91.613 |
| **std**  | 0.000 | 0.037 | 0 | 0.037 |

## Trajectory description

### Seed 42
- init=203.25, step4400=96.42, step8400=93.49, best=step16000@91.66, step12400=92.32, step16000=91.66
### Seed 137
- init=203.25, step4400=96.23, step8400=93.47, best=step16000@91.61, step12400=92.09, step16000=91.61
### Seed 2024
- init=203.25, step4400=96.27, step8400=93.12, best=step16000@91.57, step12400=92.17, step16000=91.57

## Comparison

| config | best_ppl_mean | source |
|---|---|---|
| K=1 6-block 4k steps        | 96.886 | cycle 240 |
| K=1 6-block 16k (this cycle)| **91.613 ± 0.037** | cycle 269 |
| K=1 3-block (reference)     | 59.245 | cycle 230 |
| K=512 6-block (reference)   | 59.515 | cycle 221 |
| K=128 6-block (reference)   | 67.562 | cycle 231 |
| float (reference)           | 53.895 | — |

- Δ vs cycle 240 K=1 6-block 4k: **-5.273** PPL
- Δ vs cycle 230 K=1 3-block:    **+32.368** PPL
- Δ vs cycle 221 K=512 6-block:  **+32.098** PPL
- Δ vs cycle 231 K=128 6-block:  **+24.051** PPL
- Δ vs float:                    **+37.718** PPL
