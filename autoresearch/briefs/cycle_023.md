# Cycle 023 Brief — Local Complexity as Compression Difficulty Predictor

## Item 29 Summary

Implement the Patel & Montúfar 2025 local complexity measure on a trained spirals ReLU MLP.
For each probe input x, count distinct activation patterns in a Euclidean ε-ball in 2D input space.
Average over probe points → per-layer local complexity. Correlate with single-affine residual r
from cycle 010 (Spearman ≥ 0.5 to confirm). Also compare across training checkpoints
(early=epoch 100, mid=500, late=2000) to test that local complexity decreases ≥ 30%.

## Key Context

- Cycle 010 per-layer r_fit: layer 1=0.424, layer 2=0.047, layer 3=0.031, layer 4=0.022
  (large drop from layer 1 to 2 — layer 1 is hardest to compress)
- Architecture: 2→64×4→1 spirals (n=2000, 3 turns, scale=2.0), lr=0.001
- ε ≈ 0.39 targets ~30 neighbors per ball in 2D (1000 probe points over ~16 area)
- Local complexity is computed on the 2D input activations, then propagated per layer
  (activation pattern = binary gate vector after each ReLU layer)

## Success Criteria

- Confirmed: Spearman(local_complexity, r) ≥ 0.5 AND complexity drops ≥ 30% early→late
- Refuted: no correlation with r AND no training trend
- Inconclusive: one criterion met, other fails
