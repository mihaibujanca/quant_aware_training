# Cycle 312 — 50D input eff_rank census at HIDDEN=128 (item 322)

**Verdict:** INCONCLUSIVE

## Config
- MLP(50, 128, 1, depth=12, ReLU); Adam lr=0.001; full-batch BCEWithLogits
- make_classification(n_samples=2000, n_features=50, n_informative=10, n_classes=2) + per-feature z-score
- Stop when train_acc >= 0.9 (cap 20000 epochs)
- Seeds: [42, 137, 2024]
- Trajectory epochs: [0, 50, 200, 500, 2000]
- Trainable params/net = 188289
- eff_rank = (sum sigma)^2 / sum sigma^2 of SVD(Y) (stable rank, item 322 verbatim)

## Verdict logic (item 322 verbatim)
- CONFIRMED iff (a) post mean eff_rank <= 5 at >= 8/12 layers across all 3 seeds AND (b) trajectory monotone-decreasing at >= 6/12 layers.
- REFUTED iff (c) init mean eff_rank >= 60 at all 12 layers AND (d) post mean >= 30 at all 12 layers.
- INCONCLUSIVE otherwise.

## Per-layer aggregates (mean ± std across 3 seeds)

| layer | init_mean ± std | post_mean ± std |
|------:|----------------:|----------------:|
| 0 | 69.173 ± 0.318 | 67.425 ± 0.473 |
| 1 | 50.386 ± 1.629 | 18.692 ± 0.414 |
| 2 | 31.146 ± 2.782 | 5.100 ± 0.340 |
| 3 | 13.050 ± 1.129 | 2.771 ± 0.192 |
| 4 | 4.225 ± 0.082 | 2.350 ± 0.116 |
| 5 | 1.880 ± 0.111 | 2.195 ± 0.059 |
| 6 | 1.295 ± 0.035 | 2.078 ± 0.074 |
| 7 | 1.104 ± 0.010 | 2.014 ± 0.164 |
| 8 | 1.041 ± 0.005 | 2.021 ± 0.143 |
| 9 | 1.014 ± 0.003 | 2.070 ± 0.043 |
| 10 | 1.006 ± 0.001 | 2.004 ± 0.033 |
| 11 | 1.002 ± 0.001 | 1.853 ± 0.126 |

## Trajectory (mean across 3 seeds)

| layer | e=0 | e=50 | e=200 | e=500 | e=2000 |
|------:|------:|------:|------:|------:|------:|
| 0 | 69.173 | 67.291 | 67.428 | 67.427 | 67.427 |
| 1 | 50.386 | 17.759 | 18.744 | 18.723 | 18.711 |
| 2 | 31.146 | 4.819 | 5.122 | 5.112 | 5.107 |
| 3 | 13.050 | 2.624 | 2.779 | 2.776 | 2.774 |
| 4 | 4.225 | 2.241 | 2.354 | 2.353 | 2.353 |
| 5 | 1.880 | 2.147 | 2.202 | 2.200 | 2.198 |
| 6 | 1.295 | 2.054 | 2.089 | 2.084 | 2.081 |
| 7 | 1.104 | 2.057 | 2.042 | 2.024 | 2.015 |
| 8 | 1.041 | 2.009 | 2.036 | 2.035 | 2.026 |
| 9 | 1.014 | 1.821 | 2.011 | 2.077 | 2.079 |
| 10 | 1.006 | 1.650 | 1.847 | 1.956 | 1.985 |
| 11 | 1.002 | 1.573 | 1.693 | 1.780 | 1.825 |

## Verdict clause breakdown
- (a) post eff_rank <= 5 layers (all seeds): **9/12** (need >= 8 -> True)
- (b) trajectory monotone decreasing layers: **0/12** (need >= 6 -> False)
- (c) init mean >= 60 at all 12 layers: **False**
- (d) post mean >= 30 at all 12 layers: **False**

## Per-seed convergence
- seed=42: epochs=25, train_acc=1.0000, converged=True
- seed=137: epochs=25, train_acc=1.0000, converged=True
- seed=2024: epochs=25, train_acc=1.0000, converged=True

Total wall: 291.2s
