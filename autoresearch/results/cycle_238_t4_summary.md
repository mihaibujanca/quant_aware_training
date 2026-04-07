# Cycle 238 T4 — Bump-only Activation Patching Summary

**Verdict:** REFUTED

**Config:** gpt2 / held-out wikitext-2 val / 200×128=25600 tokens / blocks=[1, 3, 5, 7, 9] / n_pairs=100 / margin_min=2.0 / positions=[32, 64, 96] / device mps

**stat_used:** median  (mean/median disagree: True)

## Per-block bump attribution

| block | n | mean | median | std | frac>0.15 | frac>0 | n_full_positive | n_null |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 0.2121 | 0.0083 | 1.4425 | 0.15 | 0.59 | 42 | 0 |
| 3 | 100 | 0.1057 | 0.0117 | 0.8980 | 0.15 | 0.60 | 54 | 0 |
| 5 | 100 | 0.0348 | -0.0093 | 0.6976 | 0.15 | 0.46 | 53 | 0 |
| 7 | 100 | 0.2132 | -0.0038 | 2.3059 | 0.15 | 0.49 | 57 | 0 |
| 9 | 100 | 2.7904 | 0.0167 | 27.1508 | 0.17 | 0.64 | 64 | 0 |

**n_blocks_mean_above_0.15 = 3/5**  |  **n_blocks_median_above_0.15 = 0/5**

## Interpretation

The centred bump residual Δ̃_l explains < 5% of the inter-token prediction shift at ALL 5 tested blocks (by median). Even though cycle 237 shows the bump subspace is load-bearing at the aggregate PPL level, at the per-prediction level the Δ̃ swap captures ~nothing of the B-token shift — the prediction difference between tokens A and B is carried by the non-Δ̃ components of x_l.

## Sanity

- Identity patch max |Δ logit| = 0.00e+00 (ok=True)
- Pool margin final = 2.0, positions final = [32, 64, 96]
- Pairs used = 100 / target 100
