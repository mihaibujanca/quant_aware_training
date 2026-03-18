# Cycle 020 Brief — Manifold Curvature Trajectory: GPT-2

## Item 38 Summary
Estimate local manifold curvature of GPT-2 hidden states at each of 13 layer
boundaries (embedding + 12 block outputs). Curvature proxy = fraction of variance
NOT explained by top-d principal components in a k=50 NN neighborhood of each token.
Average over tokens → per-layer curvature scalar. Correlate with single-block swap
damage from cycle 005.

## Hypothesis
Early layers build nonlinear curvature as the model creates distributed features.
Late layers flatten toward linear classification. Layers with high curvature are
harder to compress (more local nonlinearity to preserve).

## Criteria
- **Confirmed**: curvature decreases monotonically in final third (layers 9–12)
  AND Spearman(curvature, compression_damage) ≥ 0.4
- **Refuted**: no clear trend or anti-correlation
- **Inconclusive**: curvature decreases but Spearman 0.2–0.4

## Key Data Dependency
Cycle 005 per-layer geometric damage at 8x compression (hidden_fraction=0.125),
geometric method, latest record per layer_id.

## Execution Plan
Load GPT-2 + Shakespeare. Collect hidden states at all 13 boundaries (N≈512 tokens).
For each layer: k-NN graph (k=50), local PCA per token, estimate intrinsic dim d
(≥95% variance), curvature = unexplained variance fraction. Average. Correlate.
