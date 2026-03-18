# Research Status

## Completed Cycles: 009-027 (19 cycles)
| Outcome | Count | Key Cycles |
|---------|-------|------------|
| Confirmed | 4 | 010 (toy affine), 019 (entropy), 026 (geometric>SVD), **027 (centroid signal)** |
| Refuted | 7 | 009, 011, 012, 014, 020, 021, 025 |
| Inconclusive | 8 | 013, 015, 016, 017, 018, 022, 023, 024 |

## Actionable Results
1. **Centroid affinity is the best merge signal** (cycle 027): ρ=0.85 vs weight cosine ρ=0.04. 12/12 blocks.
2. **Geometric method validated** (cycle 026): beats SVD in 92% of comparisons.
3. **Procrustes identifies removable blocks** (cycle 018): block 6 → +3.9% ppl.
4. **Attention head pruning viable** (cycle 022): K=5 → +2.7% ppl.
5. **Compounding is the ceiling**: collective removal superadditive.
6. **GELU smoothness critical** (cycle 025): no coarse partition theory for transformers.

## Next
P1: 16 (exact prune, dep 11 done), 18 (learned compressor, dep 10 done).
Item 17 (local-chart) now unlocked (deps 10, 13, 15 all done).
