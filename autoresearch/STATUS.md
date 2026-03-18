# Research Status

## Completed Cycles: 009-024 (16 cycles)
| Outcome | Count | Key Cycles |
|---------|-------|------------|
| Confirmed | 2 | 010 (toy affine), 019 (toy entropy) |
| Refuted | 6 | 009, 011, 012, 014, 020, 021 |
| Inconclusive | 8 | 013, 015, 016, 017, 018, 022, 023, 024 |

## Central Finding
**Geometric redundancy is real but not exploitable by no-SGD compression alone.** Individual components (FFN blocks, attention heads) are demonstrably redundant by multiple metrics (Procrustes linearity, CKA, attention rank), but removing multiple simultaneously causes superadditive degradation. The compounding effect is the hard barrier.

## Next
Eligible P0: 33 (GELU PWL GPT-2, dep 32 not refuted).
P1: 14 (rank baseline), 15 (centroid affinity), 16 (exact prune), 17 (local-chart), 18 (learned compressor).
Topology items blocked (no ripser): 27, 28, 42, 43, 45.
