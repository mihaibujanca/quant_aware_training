# Research Status — 131 Cycles (001–131)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 24 |
| Refuted | 38 |
| Inconclusive | 59 |
| Crash | 2 |

## Latest Cycles
- **Cycle 131** (item 88): Bottleneck AE POC — INCONCLUSIVE (AE beats topk by 2-3.4%, promising)
- **Cycle 130** (item 87): Procrustes alignment — REFUTED (all pairs >84%)
- **Cycle 129** (item 100): FFN isotropy is learned — CONFIRMED
- **Cycle 128** (item 134): Dead neuron census — INCONCLUSIVE
- **Cycle 127** (item 144): Correction SVD alignment — INCONCLUSIVE
- **Cycle 125** (item 142): Placement sweep — REFUTED

## Key Results

### Compression Recipe
1. **FFN k≤4**: Topk + lstsq + 500 Adam at 6× → PPL ~80 (stacked)
2. **+ LoRA corrections** (r=8, 49K params): PPL ~59 (+14% over float)
3. **Bottleneck AE beats topk by 2-3.4% at k=384** (cycle 131) — promising but 2.5x param gap

### Structural
- FFN isotropy is learned (7-8x, cycle 129)
- Cascade c≈0.41 attention-driven (cycles 104, 113)

### Dead Ends
- Cross-block weight sharing (130), exact GELU pruning (128)
- Placement optimization (125), allocation (114), Jacobian (100, 103)

## Next Up (all P2)
- **126**: Attention head pruning
- **127**: Direct regression hybrid
- **135**: Data-conditioned affine map extraction
