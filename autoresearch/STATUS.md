# Research Status — 132 Cycles (001–132)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 24 |
| Refuted | 39 |
| Inconclusive | 59 |
| Crash | 2 |

## Latest Cycles
- **Cycle 132** (item 126): Attention V@O SVD pruning — REFUTED (catastrophic +243 PPL)
- **Cycle 131** (item 88): Bottleneck AE POC — INCONCLUSIVE (beats topk by 2-3.4%)
- **Cycle 130** (item 87): Procrustes alignment — REFUTED (all pairs >84%)
- **Cycle 129** (item 100): FFN isotropy is learned — CONFIRMED
- **Cycle 128** (item 134): Dead neuron census — INCONCLUSIVE

## Key Results

### Compression Recipe
1. **FFN k≤4**: Topk + lstsq + 500 Adam at 6× → PPL ~80 (stacked)
2. **+ LoRA corrections** (r=8, 49K params): PPL ~59 (+14% over float)
3. Bottleneck AE beats topk by 2-3.4% at k=384 (cycle 131, promising)

### Definitively Dead
- Attention compression: rank pruning catastrophic (cycle 132)
- Cross-block weight sharing: blocks dissimilar (cycle 130)
- Exact GELU pruning: no free compression (cycle 128)

## Next Up (P2)
- **127**: Direct regression hybrid
- **135**: Data-conditioned affine map extraction
