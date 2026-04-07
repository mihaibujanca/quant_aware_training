# Cycle 236 T2 Summary

**Verdict:** INCONCLUSIVE

**Headline:** n_blocks_pass = 0/12  |  max effect_size = 29.230

**Config:** gpt2 / held-out wikitext-2 val / 8×128=1024 tokens / calibration wikitext-2 train 200×512 / alphas=[-3, -1, 0, 1, 3] / n_rand_seeds=10 / seed 42 / device mps

## Per-block intervention effect (α=+3)

| block | max_effect | mean_effect | max_jaccard | mean_jaccard | n_hits (≥3 & ≥0.3) | rand_L2 |
|------:|-----------:|-----------:|------------:|-------------:|------------------:|---------:|
| 0 | 5.23 | 3.14 | 0.290 | 0.080 | 0/5 | 114.096 |
| 1 | 3.62 | 2.74 | 0.053 | 0.016 | 0/5 | 142.584 |
| 2 | 2.34 | 1.97 | 0.026 | 0.005 | 0/5 | 150.065 |
| 3 | 2.72 | 1.85 | 0.026 | 0.010 | 0/5 | 152.044 |
| 4 | 2.62 | 1.93 | 0.000 | 0.000 | 0/5 | 157.811 |
| 5 | 2.20 | 1.96 | 0.026 | 0.005 | 0/5 | 160.118 |
| 6 | 3.47 | 2.57 | 0.053 | 0.016 | 0/5 | 137.316 |
| 7 | 2.85 | 2.12 | 0.081 | 0.021 | 0/5 | 126.071 |
| 8 | 4.20 | 2.95 | 0.053 | 0.021 | 0/5 | 117.585 |
| 9 | 5.88 | 4.06 | 0.081 | 0.032 | 0/5 | 95.829 |
| 10 | 8.05 | 4.88 | 0.111 | 0.038 | 0/5 | 104.373 |
| 11 | 29.23 | 16.10 | 0.333 | 0.100 | 0/5 | 18.882 |

## Interpretation

Mixed result: n_blocks_pass=0/12 under the joint effect≥3 AND jaccard≥0.3 criterion. Max effect size 29.230 is distinguishable from random (29.230 ≥ 1.2), so the intervention does move the model, but the shift directions do not match T1's vocab predictions in the required number of blocks. Under an effect-only criterion (no Jaccard), 4/12 blocks pass. This is the 'correlational alignment' failure mode T1 flagged: directions are causally active but not in the way T1's wte@v decoding suggested.

## Sanity checks

- Sanity 1 (α=0 no-op): max |Δlogit| = 0.00e+00 (OK).
- Sanity 2 (linearity α=±1 ≤ α=±3): max (l2(α=1) - l2(α=3)) = -23.7511.
- Sanity 3 (std_l finite, O(1)): True (per-block mean [0.67, 0.68, 0.71, 0.69, 0.7, 0.74, 0.77, 0.81, 0.93, 1.05, 1.21, 1.42]).
- Sanity 4 (coherent counts ≥5 all blocks): confirmed from cycle 235 summary.
