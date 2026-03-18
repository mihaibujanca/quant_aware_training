# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 022 | Attention head census — GPT-2 | empirical | inconclusive | 89.6% heads low-rank; K=5 +2.7% ppl; compounding at K=10 |
| 021 | Grokking — toy | empirical | refuted | Patterns INCREASE during grokking on continuous tasks |
| 020 | Manifold curvature — GPT-2 | empirical | refuted (proxy) | Curvature flat; LayerNorm confound |
| 019 | Entropy cascade — toy | empirical | confirmed | cH drops 96% L1→L3/L4 |
| 018 | Procrustes linearity — GPT-2 | empirical | inconclusive | 7/12 layers lin≥0.98; block 6 identity +3.9% ppl |

## Meta-Finding: Compounding
The most consistent finding across ALL experiments (cycles 007, 018, 022): individual components (FFN blocks, attention heads) are geometrically redundant, but removing multiple simultaneously causes superadditive degradation. No-SGD compression has a hard ceiling imposed by this compounding effect.

## Next
Eligible P0 (ripser unavailable → topology items blocked): 29 (local complexity toy), 31 (CKA GPT-2), 33 (GELU PWL GPT-2).
P1: 14 (rank baseline), 15 (centroid affinity), 16 (exact prune), 17 (local-chart), 18 (learned compressor).

## Dead Ends (14 items)
- Tropical, stable-channel, single-affine (GPT-2), k-means/spectral on maps, intrinsic dim, coarse PWL, volume pruning, curvature (LayerNorm), grokking simplification
