# Cycle 027 Brief — Centroid and Jacobian Affinity as Merge Signals

## Item
Item 15: Centroid and Jacobian affinity as merge signals

## Question
For within-block FFN channel merging in GPT-2, which similarity signal best predicts
which channel pairs can be merged with least output damage?

## Signals to test
- **Weight cosine**: cos(w_down_i, w_down_j) — raw parameter similarity, no data.
- **Centroid similarity**: cos(μ_i, μ_j) where μ_i = mean activation of channel i over probe tokens.
- **Jacobian-spectrum similarity**: cosine similarity between per-channel Jacobian contribution
  vectors, aggregated over probe tokens.

## Protocol
For each of 12 FFN blocks:
1. Sample 200 random channel pairs from 3072 hidden channels.
2. Compute each signal for every pair.
3. Measure actual merge cost: replace pair (i,j) with weighted average channel, measure
   block output MSE increase.
4. Compute Spearman(signal, merge_cost) per block — negative correlation = better predictor.

## Criterion
Confirmed if centroid or Jacobian improves Spearman vs weight cosine by ≥0.15 on majority of blocks.
Refuted if weight cosine is equally good or better.

## Dependencies
Items 10, 11 (GPT-2 single-block compression experiments already run).
