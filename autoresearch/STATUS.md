# Research Status — 163 Cycles (001–163)

## Session 125-163 (39 cycles)

### Key Findings
1. **Topk near-optimal for channel selection** (161, 141, 143): centroid-FPS worse by 0.6 PPL, importance pruning marginal, eigenstructure worse
2. **No free pruning in GPT-2 FFN** (162, 28, 32, 83, 128): 5 cycles confirm 0.01% dead channels
3. **Error accumulates diffusely** (163): no Pareto structure in per-block degradation (max 12.6%), concentrated corrections 1.7 PPL worse than uniform
4. **Joint training exhausted** (155-158): 5 consecutive refutations
5. **Correction subspaces are compression-specific** (159): can't transfer between compressed models
6. **Topk+lstsq+corrections remains best overall**: PPL ~59 at 1.6M + 49K params

### Working Recipes (ranked)
1. **topk+lstsq + corrections** → PPL ~59 (49K correction params, total ~1.6M)
2. **topk+lstsq+polish** → PPL ~70.7 (1.58M params)

### Dead Ends (this session)
Attention pruning, cross-block sharing, exact pruning, GELU affine maps, G-weighted metrics, method hybrids, eigenstructure selection, progressive stacking, corrections on CE-trained models, higher-rank joint, joint compress+correct, warm-start joint, correction-first compression, block-adaptive ratio, centroid-affinity, GELU dead channel census, per-block error concentration

## Next
- NEXT is empty — agenda refresh required
