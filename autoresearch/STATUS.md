# Research Status — 130 Cycles (001–130)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 24 |
| Refuted | 38 |
| Inconclusive | 58 |
| Crash | 2 |

## Latest Cycles
- **Cycle 130** (item 87): Procrustes alignment — REFUTED (all pairs >84%)
- **Cycle 129** (item 100): FFN isotropy is learned — CONFIRMED (7-8x increase)
- **Cycle 128** (item 134): Dead neuron census — INCONCLUSIVE (0 strict-removable)
- **Cycle 127** (item 144): Correction SVD alignment — INCONCLUSIVE
- **Cycle 126** (item 143): PLAN INVALID
- **Cycle 125** (item 142): Placement sweep — REFUTED (P1 near-optimal)

## Key Results

### Compression Recipe
1. **FFN k≤4**: Topk + lstsq + 500 Adam at 6× → PPL ~80 (stacked)
2. **+ LoRA corrections** (r=8, 49K params): PPL ~59 (+14% over float)

### Structural
- FFN isotropy is learned (7-8x random→trained, cycle 129)
- Cascade c≈0.41 attention-driven, uniform (cycles 104, 113)
- Corrections learn partially novel subspace (cycle 127)

### Dead Ends
- Cross-block weight sharing (130): blocks completely dissimilar
- Exact GELU pruning (128): no free compression
- Placement optimization (125), allocation (114), Jacobian (100, 103)

## Next Up (all P2)
- **88**: Learned bottleneck autoencoder
- **126**: Attention head pruning
- **127**: Direct regression hybrid
- **135**: Data-conditioned affine map extraction
