# Research Status — 134 Cycles (001–134)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 24 |
| Refuted | 40 |
| Inconclusive | 60 |
| Crash | 2 |

## Latest Cycles
- **Cycle 134** (item 135): Affine map extraction — INCONCLUSIVE (no Jacobian clustering)
- **Cycle 133** (item 127): Hybrid regression — REFUTED
- **Cycle 132** (item 126): Attention pruning — REFUTED (catastrophic)
- **Cycle 131** (item 88): Bottleneck AE — INCONCLUSIVE (beats topk 2-3.4%)
- **Cycle 130** (item 87): Procrustes — REFUTED
- **Cycle 129** (item 100): Isotropy learned — CONFIRMED

## Key Results

### Working Compression Recipe
1. **FFN k≤4**: Topk + lstsq + 500 Adam at 6× → PPL ~70 post-polish
2. **+ LoRA corrections** (r=8, 49K params): PPL ~59 (+14% over float)
3. Bottleneck AE beats topk by 2-3.4% at k=384 (promising, needs follow-up)

### Definitively Dead
- GELU affine map merging (134), attention compression (132), cross-block sharing (130)
- Exact pruning (128), method hybrid (133), placement opt (125), allocation opt (114)

## Next Up
- **P0-151**: Bottleneck AE at k=512 (6x), param-matched vs topk — critical follow-up to cycle 131
- **P0-152**: AE k=4 stacking — does better local fit translate to better composition?
- **P0-145**: Curved inference census (pullback metric validation on GPT-2)
- **P0-146**: Pullback-metric error measurement vs Frobenius
- **P1-153**: Graduated importance pruning (non-binary, eigenstructure-informed)
- **P1-154**: Correction subspace analysis — what is the novel subspace corrections learn?
- **P1-155**: Eigenstructure-aware channel selection (leveraging learned isotropy, cycle 129)
- **P1-147/148**: G-weighted lstsq + curvature-guided allocation (depend on 145-146)
- **P2-156**: Nanochat asset audit and benchmark lock-in (overdue R1 direction)
- **P2-157/158**: Stacked AE + linear AE ablation (depend on 151/152)
