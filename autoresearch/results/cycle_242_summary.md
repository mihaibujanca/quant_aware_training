# Cycle 242 — Item 252: Early-only vs late-only 3-block compression

**Verdict: CONFIRMED**

- Criteria: CONFIRMED if max(Δ_A, Δ_B) ≥ 2.0; REFUTED if both |Δ| ≤ 0.8; else INCONCLUSIVE
- Float PPL (held-out, same split as cycle 210/241): **53.895**
- Cycle 210 [1,4,7] 3-block joint Adam baseline: **46.556 ± 0.028**
- Cycle 233 [3..8] 6-block baseline: 67.826
- Config: K=512, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024]
- Trainable surface per block: W_out + b_out (no W_fc / b_fc training)

## Per-config × per-seed best PPL

| config | seed | init | best | best_step | final@4000 |
|---|---|---|---|---|---|
| [1, 2, 3] | 42 | 61.695 | 47.205 | 2500 | 48.236 |
| [1, 2, 3] | 137 | 61.695 | 47.003 | 2400 | 47.747 |
| [1, 2, 3] | 2024 | 61.695 | 47.198 | 2400 | 48.080 |
| **[1, 2, 3] mean** |  | 61.695 | **47.135** |  | 48.021 |
| **[1, 2, 3] std**  |  | 0.000 | 0.093 |  | 0.204 |
| [8, 9, 10] | 42 | 71.349 | 55.173 | 1400 | 62.367 |
| [8, 9, 10] | 137 | 71.349 | 55.248 | 1400 | 62.216 |
| [8, 9, 10] | 2024 | 71.349 | 55.155 | 1400 | 62.159 |
| **[8, 9, 10] mean** |  | 71.349 | **55.192** |  | 62.247 |
| **[8, 9, 10] std**  |  | 0.000 | 0.040 |  | 0.088 |

## Comparison to baselines

| config | mean_best ± std | Δ vs [1,4,7]=46.556 |
|---|---|---|
| [1, 2, 3] | 47.135 ± 0.093 | +0.579 |
| [8, 9, 10] | 55.192 ± 0.040 | +8.636 |

## Verdict rationale

- max(Δ_A, Δ_B) = max(+0.579, +8.636) = +8.636
- |Δ_A| = 0.579, |Δ_B| = 8.636
- **CONFIRMED**: at least one config is ≥ 2.0 PPL worse than 46.556 → depth placement matters, not just block count.
