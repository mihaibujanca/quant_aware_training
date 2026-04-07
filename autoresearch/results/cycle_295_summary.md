# Cycle 295 — Spirals/Moons eff_rank census (item 305)

**Verdict:** INCONCLUSIVE

## Config
- MLP(2, 32, 1, depth=12, ReLU); Adam lr=0.001; full-batch BCEWithLogits
- N=2000; spirals(noise=0.5, n_turns=3); moons(noise=0.1)
- Stop when train_acc ≥ 0.95 (cap 20000 epochs)
- Seeds: [42, 137, 2024]
- Trainable params/net = 11745

## Verdict logic (item 305 verbatim)
- CONFIRMED iff post eff_rank ≤ 3 in ≥8 of 12 layers AND init eff_rank ≥ 20 in ≥8 of 12 layers
- REFUTED iff post eff_rank ≥ 10 in ≥6 layers
- INCONCLUSIVE otherwise

## spirals (per-dataset verdict: INCONCLUSIVE)

| layer | init_mean ± std | post_mean ± std |
|------:|----------------:|----------------:|
| 0 | 2.994 ± 0.239 | 3.143 ± 0.213 |
| 1 | 2.705 ± 0.274 | 2.859 ± 0.192 |
| 2 | 1.874 ± 0.235 | 2.106 ± 0.156 |
| 3 | 1.187 ± 0.079 | 2.088 ± 0.090 |
| 4 | 1.058 ± 0.022 | 2.151 ± 0.179 |
| 5 | 1.010 ± 0.003 | 1.937 ± 0.157 |
| 6 | 1.002 ± 0.000 | 1.986 ± 0.225 |
| 7 | 1.001 ± 0.000 | 2.072 ± 0.299 |
| 8 | 1.000 ± 0.000 | 2.049 ± 0.125 |
| 9 | 1.000 ± 0.000 | 2.032 ± 0.368 |
| 10 | 1.000 ± 0.000 | 1.756 ± 0.276 |
| 11 | 1.000 ± 0.000 | 1.736 ± 0.122 |

- post layers with eff_rank ≤ 3: **11/12** (need ≥8 for CONFIRMED)
- init layers with eff_rank ≥ 20: **0/12** (need ≥8 for CONFIRMED)
- post layers with eff_rank ≥ 10: **0/12** (REFUTED iff ≥6)

## moons (per-dataset verdict: INCONCLUSIVE)

| layer | init_mean ± std | post_mean ± std |
|------:|----------------:|----------------:|
| 0 | 2.179 ± 0.269 | 2.153 ± 0.223 |
| 1 | 2.181 ± 0.159 | 1.463 ± 0.019 |
| 2 | 1.645 ± 0.051 | 1.234 ± 0.041 |
| 3 | 1.154 ± 0.073 | 1.178 ± 0.067 |
| 4 | 1.052 ± 0.022 | 1.204 ± 0.115 |
| 5 | 1.009 ± 0.004 | 1.300 ± 0.165 |
| 6 | 1.001 ± 0.000 | 1.406 ± 0.215 |
| 7 | 1.000 ± 0.000 | 1.520 ± 0.293 |
| 8 | 1.000 ± 0.000 | 1.611 ± 0.300 |
| 9 | 1.000 ± 0.000 | 1.503 ± 0.217 |
| 10 | 1.000 ± 0.000 | 1.358 ± 0.135 |
| 11 | 1.000 ± 0.000 | 1.405 ± 0.289 |

- post layers with eff_rank ≤ 3: **12/12** (need ≥8 for CONFIRMED)
- init layers with eff_rank ≥ 20: **0/12** (need ≥8 for CONFIRMED)
- post layers with eff_rank ≥ 10: **0/12** (REFUTED iff ≥6)

## Per-seed convergence
- spirals seed=42: epochs=11325, train_acc=0.9520, converged=True
- spirals seed=137: epochs=9625, train_acc=0.9505, converged=True
- spirals seed=2024: epochs=6075, train_acc=0.9510, converged=True
- moons seed=42: epochs=75, train_acc=0.9510, converged=True
- moons seed=137: epochs=100, train_acc=0.9990, converged=True
- moons seed=2024: epochs=100, train_acc=0.9985, converged=True

Total wall: 62.2s
