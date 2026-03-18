# Research Status — 20 Cycles Complete (009-028)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 4 (010, 019, 026, 027) |
| Refuted | 8 (009, 011, 012, 014, 020, 021, 025, 028) |
| Inconclusive | 8 (013, 015, 016, 017, 018, 022, 023, 024) |

## Answer to the Research Question
"Given a trained transformer LM, how do we construct a smaller FFN reparameterization that preserves geometry without SGD?"

**Partial answer after 20 cycles:**
- **Single-component removal works**: Block 6 identity → +3.9% ppl (cycle 018). 5 attention heads → +2.7% ppl (cycle 022).
- **Data-driven channel selection works**: Geometric method beats SVD by consistent margin (cycle 026). Centroid activation similarity (ρ=0.85) is the right merge signal (cycle 027).
- **Structural pruning is empty**: No dead/linear channels in GELU FFNs (cycles 006, 028).
- **Compounding is the hard ceiling**: Removing multiple components is superadditive (cycles 007, 018, 022). No-SGD compression alone gives ~1 block + 5 heads.
- **Partition theory doesn't transfer**: GELU needs ≥20 PWL pieces (cycle 025). k-means/spectral on maps fail (cycles 012-014).

## Next
P1: 17 (local-chart, deps met), 18 (learned compressor, dep 10 done).
