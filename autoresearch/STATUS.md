# Research Status — 95 Cycles (001–095)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 15 |
| Refuted | 27 |
| Inconclusive | 44 |
| Crash | 1 |

## Latest Cycle
- **Cycle 095** (item 109): Attention compression v2 — GPT-2 — INCONCLUSIVE
  - 5/12 blocks tolerate 5.4× QKV compression at ≤+5% PPL (needed ≥6 for confirmed)
  - Block 0 catastrophically sensitive (+102%) — early attention is critical
  - rank-16 ≈ rank-11 in multi-block: same inter-block propagation bottleneck as FFN
  - Attention compression is partially viable; needs block-0-skip strategy

## Key Structural Finding (unified across FFN + attention)
Inter-block error propagation dominates per-block approximation quality:
- FFN: c≈0.41 ratio-independent (093), allocation-independent (094)
- Attention: rank-16 ≈ rank-11 in stacking (095)
- The compression ceiling is set by cascade dynamics, not per-block fidelity

## The Recipe (updated)
1. **FFN**: Topk + lstsq + 500 Adam steps, k≤4 blocks at 6× → ~+5% PPL
2. **Attention**: Per-head rank-11 SVD on QKV (skip block 0), 5/12 blocks viable
3. Stacking limit is structural ~41%/block cascade amplification

## Remaining NEXT Items
- **P0**: Item 103 (low-rank Jacobian)
- **P2**: Items 87-88 (Procrustes, learned bottleneck AE)
- Only 3 items — agenda refresh required
