# Research Status — 129 Cycles (001–129)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 24 |
| Refuted | 37 |
| Inconclusive | 58 |
| Crash | 2 |

## Latest Cycles
- **Cycle 129** (item 100): FFN isotropy is learned — CONFIRMED
  - Random-init ratio ~0.002, trained ~0.015 (7-8x increase, blocks 4-7)
  - Block 1 anomalous (1.5x only), GELU→ReLU swap negligible effect
- **Cycle 128** (item 134): Dead/always-active neuron census — INCONCLUSIVE
  - 0/36864 strictly removable; exact pruning dead end for GELU
- **Cycle 127** (item 144): Correction SVD alignment — INCONCLUSIVE
- **Cycle 125** (item 142): Placement sweep — REFUTED

## Key Results

### Compression Recipe
1. **FFN k≤4**: Topk + lstsq + 500 Adam at 6× → PPL ~80 (stacked)
2. **+ LoRA corrections** (r=8, 49K params): PPL ~59 (+14% over float)
3. Placement near-optimal at residual stream (cycle 125)
4. Corrections learn partially novel subspace (cycle 127)

### Structural Findings
- FFN isotropy is learned (7-8x random→trained, cycle 129)
- Cascade c≈0.41 attention-driven, uniform (cycles 104, 113)
- GELU negative leak = 87-99% constant bias (cycle 109)

### Dead Ends
- Exact GELU pruning (128), allocation optimization (114), Jacobian methods (100, 103)
- Correction placement (125), GELU clustering (101), head targeting (113)

## Next Up
- **87**: Cross-block Procrustes alignment (P2)
- **88**: Learned bottleneck autoencoder (P2)
- **127**: Direct regression hybrid (P2)
- **135**: Data-conditioned affine map extraction (P2)
- **126**: Attention head pruning (P2)
