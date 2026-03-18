# Research Status

## Completed Cycles: 009-023 (15 cycles)
| Outcome | Count | Key Cycles |
|---------|-------|------------|
| Confirmed | 2 | 010 (toy single-affine), 019 (toy entropy cascade) |
| Refuted | 6 | 009, 011, 012, 014, 020, 021 |
| Inconclusive | 7 | 013, 015, 016, 017, 018, 022, 023 |

## Strongest Signals
1. **Procrustes linearity** (cycle 018): 7/12 GPT-2 layers near-identity. Block 6 removable at +3.9% ppl.
2. **Attention head low-rank** (cycle 022): 89.6% heads low-rank. K=5 prunable at +2.7% ppl.
3. **Entropy cascade** (cycle 019): Deep toy layers add zero new partition information.
4. **Compounding is the hard problem**: Individual removal is cheap, but collective removal is superadditive.

## Next
Eligible P0: 31 (CKA GPT-2), 33 (GELU PWL GPT-2).
Eligible P1: 14, 15, 16, 17, 18 (core compression experiments).
Topology items blocked (ripser unavailable): 27, 28, 42, 43, 45.
