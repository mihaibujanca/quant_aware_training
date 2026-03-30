# Research Status — 128 Cycles (001–128)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 23 |
| Refuted | 37 |
| Inconclusive | 58 |
| Crash | 2 |

## Latest Cycles
- **Cycle 128** (item 134): Dead/always-active neuron census — INCONCLUSIVE
  - 0/36864 strictly removable GELU neurons; 14 soft95 dead in blocks 0-4
  - Exact pruning (Kumar/Serra/Berzins) is a dead end for GELU networks
- **Cycle 127** (item 144): Correction SVD alignment — INCONCLUSIVE
  - Corrections learn partially novel subspace; moderate error overlap (0.44-0.46)
- **Cycle 126** (item 143): PLAN INVALID — predecessor 142 refuted
- **Cycle 125** (item 142): Placement sweep — REFUTED (P1 near-optimal)

## Key Results (consolidated)

### Cascade Mechanism & Correction
- c≈0.41 attention-driven, uniform across heads (cycles 104, 113)
- **LoRA corrections recover 74%** degradation, 49K params (cycle 115/116)
- Placement near-optimal at residual stream (cycle 125)
- Corrections learn partially novel subspace (cycle 127)

### Compression Recipe
1. **FFN k≤4**: Topk + lstsq + 500 Adam at 6× → PPL ~80 (stacked)
2. **+ LoRA corrections** (r=8, 49K params): PPL ~59 (+14% over float)
3. **FFN k=5**: +22-24% polished (cycle 106)

### Dead Ends
- Exact pruning for GELU (cycle 128): no free compression
- Allocation optimization (114), Jacobian methods (100, 103), GELU clustering (101)
- Correction placement (125): residual stream already optimal

## Next Up
- **100**: Is FFN isotropy architectural or learned? (P1, depends on 99)
- **87**: Cross-block Procrustes alignment (P2)
- **88**: Learned bottleneck autoencoder (P2)
- **127**: Direct regression hybrid (P2)
- **135**: Data-conditioned affine map extraction (P2)
