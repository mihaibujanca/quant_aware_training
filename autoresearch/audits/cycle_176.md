# Cycle 176 Audit — Item 183

## Verdict: REFUTED

K=8 piecewise lstsq PPL 193.0 vs K=1 baseline 61.6 — 131 points WORSE.
Item criterion: "< 1 PPL improvement" triggers REFUTED. This exceeds that threshold
by two orders of magnitude in the wrong direction.

## Guard checks

G1_implementation=PASS
G2_seeds=PASS (2 complete seeds [42, 137], seed 7 partial [K=1, K=4 done, K=8 interrupted])
G3_interpretation=PASS
G4_counterintuitive=N/A (result matches prediction from cycle 167)
G5_learning=PASS (500 Adam polish per config, loss decreasing normally)
G6_method_failure=PASS

## Key metrics

| K  | Mean PPL (polished) | Delta vs K=1 | Seed 42  | Seed 137 |
|----|---------------------|--------------|----------|----------|
| 1  | 61.6                | 0            | 70.8     | 52.3     |
| 4  | 74.3                | +12.7        | 87.4     | 61.2     |
| 8  | 193.0               | +131.5       | 298.7    | 87.3     |
| 16 | 137.3               | +75.7        | 99.4     | 175.1    |

Teacher PPL: 52.3 (mean of 3 seeds).

## Analysis

The failure is straightforward: with 4096 probe tokens split into K clusters, each
cluster gets 256-1000 tokens. A per-cluster lstsq with 512 input features and 768
output dimensions has 393K free parameters — far exceeding the sample count. The
result is massive overfitting that produces projections that fail at inference.

The non-monotonic K=16 < K=8 pattern (for seed 42) is a random artifact of cluster
quality, not a real signal — seed 137 shows the reverse (K=16 worse than K=8).

This result was predicted by cycle 167's census: sign patterns have no natural
clusters, so forced clustering creates arbitrary partitions that harm fitting.

## Completeness note

Seed 7 was interrupted during K=8 polish (process killed by timeout). Its partial
data (K=1: 55.0, K=4: 61.5) is consistent with the pattern. Two complete seeds
are sufficient for an unambiguous verdict given the magnitude of the effect.
