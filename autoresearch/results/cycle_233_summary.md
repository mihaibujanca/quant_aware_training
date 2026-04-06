# Cycle 233 — Item 243: Alternative 6-block subsets

**Verdict: INCONCLUSIVE**

- Criteria: Confirmed = any subset best_mean ≤ 58.0; Refuted = all three ∈ [59.0, 62.0]; else Inconclusive
- Config: K=512, LR=0.0001, N_STEPS=4000, EVAL_EVERY=100, SEEDS_P1=[42], SEEDS_P2=[42, 137, 2024]
- Float PPL: 53.895
- Cycle 221 baseline [1,2,4,5,7,8] best_mean: 59.515

## Per-subset results

| subset | init (s42) | phase1 best (s42) | phase2 best_mean | Δ vs c221 |
|---|---|---|---|---|
| [1, 2, 3, 4, 5, 6] | 87.631 | 61.785 | — | +2.270 |
| [3, 4, 5, 6, 7, 8] | 96.758 | 67.826 | — | +8.311 |
| [1, 3, 5, 7, 8, 10] | 87.237 | 60.981 | — | +1.466 |
| [1,2,4,5,7,8] (c221) | 86.686 | — | 59.515 ± 0.053 | 0.000 |

## Phase 1 trajectories (seed 42)

- **[1, 2, 3, 4, 5, 6]**: init=87.63, step1100=62.21, step2100=63.66, best=step1300@61.78, step3100=68.27, step4000=76.96
- **[3, 4, 5, 6, 7, 8]**: init=96.76, step1100=67.83, step2100=70.38, best=step1100@67.83, step3100=76.55, step4000=86.27
- **[1, 3, 5, 7, 8, 10]**: init=87.24, step1100=61.04, step2100=62.81, best=step1400@60.98, step3100=68.78, step4000=75.47

## Interpretation

Mixed/inconclusive — results span wider than the decision bands.
