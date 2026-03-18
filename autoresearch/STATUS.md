# Research Status

## Completed Cycles: 009-026 (18 cycles)
| Outcome | Count | Key Cycles |
|---------|-------|------------|
| Confirmed | 3 | 010 (toy affine), 019 (entropy), **026 (geometric > SVD)** |
| Refuted | 7 | 009, 011, 012, 014, 020, 021, 025 |
| Inconclusive | 8 | 013, 015, 016, 017, 018, 022, 023, 024 |

## Key Results
1. **Geometric method validated** (cycle 026): data-driven channel selection beats SVD in 92% of comparisons. Naive rank reduction is catastrophically broken for GELU FFNs.
2. **Procrustes linearity** (cycle 018): 7/12 layers near-identity. Block 6 removable at +3.9% ppl.
3. **Attention head pruning** (cycle 022): K=5 heads → +2.7% ppl.
4. **Compounding barrier**: individual removal cheap, collective removal superadditive.
5. **GELU smoothness critical** (cycle 025): partition theory requires ≥20 PWL pieces on GPT-2.

## Next
P1: 15 (centroid/Jacobian affinity, deps 10+11 done), 16 (exact prune, dep 11 done), 18 (learned compressor, dep 10 done).
Item 17 (local-chart) depends on 10, 13, 15 — 15 not done yet.
