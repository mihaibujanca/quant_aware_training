# Research Status — 161 Cycles (001–161)

## Session 125-161 (37 cycles)

### Key Findings
1. **Topk near-optimal for channel selection** (161, 141, 143): centroid-FPS worse by 0.6 PPL, importance pruning marginal (+0.2-2%), eigenstructure worse
2. **Joint training exhausted** (155-158): random-init 91.1, warm-start 73.0, polish 70.7 — 5 refutations
3. **Correction subspaces are compression-specific** (159): can't transfer between compressed models
4. **Corrections are cascade compensation** (151): 93-97% novel subspace
5. **Topk+lstsq+corrections remains best overall**: PPL ~59 at 1.6M + 49K params

### Working Recipes (ranked)
1. **topk+lstsq + corrections** → PPL ~59 (49K correction params, total ~1.6M)
2. **topk+lstsq+polish** → PPL ~70.7 (1.58M params)

### Dead Ends (this session)
Attention pruning, cross-block sharing, exact pruning, GELU affine maps, G-weighted metrics, method hybrids, eigenstructure selection, progressive stacking, corrections on CE-trained models, higher-rank joint, joint compress+correct, warm-start joint, correction-first compression, block-adaptive ratio (duplicate), centroid-affinity channel selection

## Next
- **172**: GELU-aware dead/saturated channel census (P2) — R2 exact white-box baseline
- **173**: Per-block error contribution profiling (P2) — identify cascade error sources
