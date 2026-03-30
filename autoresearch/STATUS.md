# Research Status — 133 Cycles (001–133)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 24 |
| Refuted | 40 |
| Inconclusive | 59 |
| Crash | 2 |

## Latest Cycles
- **Cycle 133** (item 127): Hybrid regression — REFUTED (topk beats hybrid post-polish)
- **Cycle 132** (item 126): Attention pruning — REFUTED (catastrophic degradation)
- **Cycle 131** (item 88): Bottleneck AE — INCONCLUSIVE (beats topk by 2-3.4%)
- **Cycle 130** (item 87): Procrustes — REFUTED (all pairs >84%)
- **Cycle 129** (item 100): Isotropy is learned — CONFIRMED
- **Cycle 128** (item 134): Dead neuron census — INCONCLUSIVE

## Key Results

### Compression Recipe
1. **FFN k≤4**: Topk + lstsq + 500 Adam at 6× → PPL ~80 → ~70 post-polish
2. **+ LoRA corrections** (r=8, 49K params): PPL ~59 (+14% over float)
3. Bottleneck AE beats topk by 2-3.4% at k=384 (cycle 131, promising)

### Dead Ends (definitive)
- Attention compression (132), cross-block sharing (130), exact pruning (128)
- Method-adaptive hybrid (133), placement optimization (125), allocation (114)

## Next Up
- **135**: Data-conditioned affine map extraction (last P2 item)
- After 135: agenda refresh needed
