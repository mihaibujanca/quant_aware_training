# Cycle 237 T3 — Bump Subspace Ablation Summary

**Verdict:** CONFIRMED

**Config:** gpt2 / held-out wikitext-2 val / 200×512=102400 tokens / k_bump=10 / random seeds [42, 137, 2024] / device mps

## Joint ablation (all 12 blocks)

| condition | PPL | ratio vs baseline | ratio vs random |
|---|---:|---:|---:|
| baseline | 34.514 | 1.000 | — |
| bump k=10 | 2344385.825 | 67925.666 | 23143.383 |
| random k=10 (mean±std, n=3) | 101.298 ± 8.137 | 2.935 | 1.000 |
| full-FFN top-10 | 3114154.919 | 90228.769 | 30742.414 |

## Per-block single ablation

| block | bump | random mean±std | full-FFN | bump/random | above 2× |
|---:|---:|---:|---:|---:|---:|
| 0 | 690.178 | 34.682 ± 0.194 | 134.369 | 19.900 | YES |
| 1 | 453.020 | 35.176 ± 0.180 | 197.909 | 12.878 | YES |
| 2 | 396.673 | 37.459 ± 1.164 | 47.082 | 10.589 | YES |
| 3 | 1122.504 | 37.228 ± 1.574 | 11316.963 | 30.152 | YES |
| 4 | 833.253 | 37.746 ± 0.277 | 71.267 | 22.075 | YES |
| 5 | 953.203 | 37.866 ± 0.670 | 112.866 | 25.173 | YES |
| 6 | 30603.616 | 38.367 ± 1.792 | 182.743 | 797.659 | YES |
| 7 | 437.063 | 38.908 ± 2.452 | 152.709 | 11.233 | YES |
| 8 | 60.950 | 36.943 ± 0.591 | 95.211 | 1.650 |  |
| 9 | 51.442 | 35.587 ± 0.362 | 68.672 | 1.446 |  |
| 10 | 53.818 | 35.599 ± 0.259 | 54.965 | 1.512 |  |
| 11 | 33758.978 | 36.362 ± 0.562 | 4399.784 | 928.410 | YES |

**n_blocks_above_2x_random = 9 / 12**

## Interpretation

The top-10 centred-Δ̃ bump subspace is significantly more load-bearing for GPT-2's residual stream than a random k=10 subspace: joint ablation inflates PPL by 23143.38× the random baseline, and 9/12 single-block ablations reach ≥2× their random control. The bump directions are not compression slack — ablating them degrades held-out LM performance above chance.

## Sanity

- V_0.T @ V_0 - I max |·| = 1.07e-06
- P_0 @ P_0 - P_0 max |·| = 5.07e-07
- Baseline before/after hooks: 34.513991 / 34.513991 (|Δ|=0.00e+00)
