# Research Status

## Completed Cycles: 009-025 (17 cycles)
| Outcome | Count | Key Cycles |
|---------|-------|------------|
| Confirmed | 2 | 010 (toy affine), 019 (toy entropy) |
| Refuted | 7 | 009, 011, 012, 014, 020, 021, 025 |
| Inconclusive | 8 | 013, 015, 016, 017, 018, 022, 023, 024 |

## Central Conclusions (17 cycles of evidence)

### What works
1. **Procrustes linearity** identifies removable blocks: Block 6 → +3.9% ppl (cycle 018)
2. **Attention head pruning**: K=5 heads → +2.7% ppl (cycle 022)
3. **Toy networks linearize** deep layers (cycles 010, 019) — but this doesn't transfer to GPT-2

### What doesn't work
1. **Single-affine collapse** for GPT-2: all blocks too nonlinear (cycles 009, 011)
2. **Affine map clustering** (k-means, spectral): no exploitable redundancy (cycles 012-014)
3. **GELU → PWL at coarse granularity**: catastrophic on GPT-2 (cycle 025). Need ≥20 pieces.
4. **Compounding is the hard barrier**: individual removal is cheap, collective removal is superadditive (cycles 007, 018, 022)

### Key insight
No-SGD geometric compression is viable for SINGLE-COMPONENT removal (1 block or ~5 heads), but the compounding effect imposes a hard ceiling. For deeper compression, learned/SGD-based methods are necessary.

## Next
P1 items: 14 (rank baseline), 15 (centroid affinity), 16 (exact prune), 17 (local-chart), 18 (learned compressor).
All P0 items exhausted (except topology items blocked by missing ripser).
