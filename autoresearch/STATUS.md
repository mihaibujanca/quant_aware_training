# Research Status — 162 Cycles (001–162)

## Session 125-162 (38 cycles)

### Key Findings
1. **Topk near-optimal for channel selection** (161, 141, 143): centroid-FPS worse by 0.6 PPL, importance pruning marginal, eigenstructure worse
2. **No free pruning in GPT-2 FFN** (162, 28, 32, 83, 128): 5 cycles confirm 0.01% dead channels, no removable channels via GELU thresholding
3. **Joint training exhausted** (155-158): 5 consecutive refutations across all initialization strategies
4. **Correction subspaces are compression-specific** (159): can't transfer between compressed models
5. **Topk+lstsq+corrections remains best overall**: PPL ~59 at 1.6M + 49K params

### Working Recipes (ranked)
1. **topk+lstsq + corrections** → PPL ~59 (49K correction params, total ~1.6M)
2. **topk+lstsq+polish** → PPL ~70.7 (1.58M params)

### Dead Ends (this session)
Attention pruning, cross-block sharing, exact pruning, GELU affine maps, G-weighted metrics, method hybrids, eigenstructure selection, progressive stacking, corrections on CE-trained models, higher-rank joint, joint compress+correct, warm-start joint, correction-first compression, block-adaptive ratio, centroid-affinity, GELU dead channel census (5th refutation)

## Next
- **173**: Per-block error contribution profiling (P2) — identify cascade error sources
- After 173: NEXT will be empty → agenda refresh required
