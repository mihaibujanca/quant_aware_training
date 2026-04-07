# Cycle 244 — Item 254: Sliding 3-block sweep

**Verdict: INCONCLUSIVE**

- Criteria: CONFIRMED if any window mean_best ≤ 48.5; REFUTED if all ≥ 52.0; else INCONCLUSIVE
- Float PPL (held-out, same split as cycles 210/241/242): **53.895**
- Cycle 210 [1,4,7] 3-block joint Adam baseline: **46.556 ± 0.028**
- Cycle 242 [1,2,3] early anchor: **47.135 ± 0.093**
- Cycle 242 [8,9,10] late anchor: **55.192 ± 0.04**
- Config: K=512, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Trainable surface per block: W_out + b_out (no W_fc / b_fc training)

## Per-config × per-seed best PPL

| config | seed | init | best | best_step | final@4000 |
|---|---|---|---|---|---|
| [3, 4, 5] | 42 | 65.853 | 49.411 | 2500 | 50.534 |
| [3, 4, 5] | 137 | 65.853 | 49.483 | 2400 | 50.213 |
| [3, 4, 5] | 2024 | 65.853 | 49.538 | 2500 | 50.265 |
| **[3, 4, 5] mean** |  | 65.853 | **49.477** |  | 50.337 |
| **[3, 4, 5] std**  |  | 0.000 | 0.052 |  | 0.141 |
| [5, 6, 7] | 42 | 66.538 | 50.871 | 2000 | 53.505 |
| [5, 6, 7] | 137 | 66.538 | 50.860 | 2100 | 52.987 |
| [5, 6, 7] | 2024 | 66.538 | 50.964 | 1400 | 53.437 |
| **[5, 6, 7] mean** |  | 66.538 | **50.899** |  | 53.310 |
| **[5, 6, 7] std**  |  | 0.000 | 0.047 |  | 0.230 |
| [6, 7, 8] | 42 | 67.651 | 52.068 | 1500 | 56.542 |
| [6, 7, 8] | 137 | 67.651 | 52.025 | 1600 | 56.088 |
| [6, 7, 8] | 2024 | 67.651 | 52.022 | 1600 | 56.031 |
| **[6, 7, 8] mean** |  | 67.651 | **52.038** |  | 56.220 |
| **[6, 7, 8] std**  |  | 0.000 | 0.021 |  | 0.229 |

## Comparison to baselines / anchors

| config | mean_best ± std | Δ vs [1,4,7]=46.556 | Δ vs c242 [1,2,3]=47.135 | Δ vs c242 [8,9,10]=55.192 |
|---|---|---|---|---|
| [3, 4, 5] | 49.477 ± 0.052 | +2.921 | +2.342 | -5.715 |
| [5, 6, 7] | 50.899 ± 0.047 | +4.343 | +3.764 | -4.293 |
| [6, 7, 8] | 52.038 ± 0.021 | +5.482 | +4.903 | -3.154 |

## Verdict rationale

- [3, 4, 5] mean_best = 49.477
- [5, 6, 7] mean_best = 50.899
- [6, 7, 8] mean_best = 52.038
- any ≤ 48.5? False; all ≥ 52.0? False
- **INCONCLUSIVE**: windows straddle the 48.5–52.0 band.
