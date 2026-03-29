# Research Status — 107 Cycles (001–107)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 17 |
| Refuted | 33 |
| Inconclusive | 48 |
| Crash | 1 |

## Latest Cycle
- **Cycle 107** (item 117): Output-space direct regression — INCONCLUSIVE (audit corrected)
  - Direct rank-r regression vs topk+lstsq at matched params
  - Middle blocks (4, 7): near-parity (direct wins 0.1-0.5%, within noise)
  - Early/late blocks (1, 11): topk wins by 2-27%
  - Audit flagged: topk pseudo-replication, margins within noise
  - Key finding: block-dependent linearity gradient

## Key Structural Findings (consolidated)

### Cascade Mechanism (cycle 104)
- **c≈0.41 is attention-driven**, not GELU-nonlinearity-driven
- GELU attenuates error in early layers, pivots at T6-T7
- Intervention target: attention error redistribution

### Compression Recipe (updated)
1. **FFN**: Topk + lstsq + 500 Adam, k≤4 blocks at 6× → ~+5% polished (k=1)
2. **FFN k=5**: +22-24% polished (cycle 106), median/best seed can hit +10%
3. **Attention**: Per-head rank-11 SVD on QKV (skip block 0), 5/12 individual viable
4. **Joint**: k=2+k=2 is the practical limit (~+19%)

### Block Linearity Gradient (cycle 107)
- Middle blocks (4, 7): nearly linear, direct regression ≈ topk
- Early (1) and late (11): strongly nonlinear, topk exploits hidden structure

### Dead Directions (cycles 099–103)
- Subspace-projection correction, Jacobian factorization, GELU' clustering, zero-shot GELU→ReLU, GELU bump masking — all ruled out

## Remaining NEXT Items
- **P1**: Item 100 (blocked on 99)
- **P2**: Items 87, 88, 118 (cascade corrections)
- 3 eligible items — AGENDA REFRESH NEEDED
