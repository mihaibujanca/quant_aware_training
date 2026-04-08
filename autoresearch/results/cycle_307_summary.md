# Cycle 307 — 50D input eff_rank census (item 317)

**Verdict:** INCONCLUSIVE

## Config
- MLP(50, 32, 1, depth=12, ReLU); Adam lr=0.001; full-batch BCEWithLogits
- make_classification(n_samples=2000, n_features=50, n_informative=10, n_classes=2) + per-feature z-score
- Stop when train_acc >= 0.9 (cap 20000 epochs)
- Seeds: [42, 137, 2024]
- Trajectory epochs: [0, 50, 200, 500, 2000]
- Trainable params/net = 13281

## Verdict logic (item 317 verbatim)
- CONFIRMED iff (a) init mean eff_rank >= 15 at layers 1-6 across all 3 seeds AND (b) post-training mean eff_rank <= 5 at >= 8 of 12 layers across all 3 seeds AND (c) trajectory monotone decreasing at >= 6 of 12 layers.
- REFUTED iff (a) passes but post-training eff_rank <= 5 at < 4 of 12 layers.
- INCONCLUSIVE otherwise.

## Per-layer aggregates (mean ± std across 3 seeds)

| layer | init_mean ± std | post_mean ± std |
|------:|----------------:|----------------:|
| 0 | 15.698 ± 0.446 | 13.475 ± 0.561 |
| 1 | 5.926 ± 1.005 | 2.370 ± 0.144 |
| 2 | 2.963 ± 0.385 | 1.745 ± 0.131 |
| 3 | 1.349 ± 0.077 | 1.741 ± 0.052 |
| 4 | 1.051 ± 0.007 | 1.767 ± 0.237 |
| 5 | 1.009 ± 0.002 | 1.767 ± 0.284 |
| 6 | 1.002 ± 0.000 | 1.712 ± 0.351 |
| 7 | 1.000 ± 0.000 | 1.677 ± 0.381 |
| 8 | 1.000 ± 0.000 | 1.551 ± 0.302 |
| 9 | 1.000 ± 0.000 | 1.524 ± 0.190 |
| 10 | 1.000 ± 0.000 | 1.547 ± 0.097 |
| 11 | 1.000 ± 0.000 | 1.743 ± 0.145 |

## Trajectory (mean across 3 seeds)

| layer | e=0 | e=50 | e=200 | e=500 | e=2000 |
|------:|------:|------:|------:|------:|------:|
| 0 | 15.698 | 14.122 | 13.468 | 13.473 | 13.474 |
| 1 | 5.926 | 3.404 | 2.389 | 2.371 | 2.370 |
| 2 | 2.963 | 2.179 | 1.754 | 1.746 | 1.745 |
| 3 | 1.349 | 1.883 | 1.732 | 1.740 | 1.738 |
| 4 | 1.051 | 1.735 | 1.722 | 1.764 | 1.764 |
| 5 | 1.009 | 1.571 | 1.705 | 1.769 | 1.767 |
| 6 | 1.002 | 1.565 | 1.691 | 1.734 | 1.716 |
| 7 | 1.000 | 1.601 | 1.701 | 1.721 | 1.689 |
| 8 | 1.000 | 1.599 | 1.560 | 1.603 | 1.568 |
| 9 | 1.000 | 1.418 | 1.417 | 1.582 | 1.548 |
| 10 | 1.000 | 1.374 | 1.403 | 1.571 | 1.546 |
| 11 | 1.000 | 1.283 | 1.479 | 1.697 | 1.708 |

## Verdict clause breakdown
- (a) init eff_rank >= 15 at layers 1-6 per-seed: **0/18** (need 18/18 → False)
- (b) layers with post eff_rank <= 5 across all seeds: **11/12** (need >= 8 → True)
- (c) trajectory monotone decreasing layers: **2/12** (need >= 6 → False)

## Per-seed convergence
- seed=42: epochs=50, train_acc=1.0000, converged=True
- seed=137: epochs=50, train_acc=1.0000, converged=True
- seed=2024: epochs=50, train_acc=1.0000, converged=True

Total wall: 138.1s
