# Research Status — 172 Cycles (001–172)

## Session 125-172 (48 cycles)

### Key Findings
1. **Topk near-optimal for channel selection** (161, 141, 143): centroid-FPS worse by 0.6 PPL, importance pruning marginal, eigenstructure worse
2. **No free pruning in GPT-2 FFN** (162, 28, 32, 83, 128): 5 cycles confirm 0.01% dead channels
3. **Error accumulates diffusely** (163): no Pareto structure, concentrated corrections 1.7 PPL worse
4. **Joint training exhausted** (155-158): 5 consecutive refutations
5. **Correction subspaces are compression-specific** (159): can't transfer between compressed models
6. **Topk+lstsq+corrections remains best overall**: PPL ~59 at 1.6M + 49K params
7. **Lstsq target is already optimal** (172): norm-weighting, SVD-subspace, and token-frequency alternatives all within 1 PPL or worse

### Working Recipes (ranked)
1. **topk+lstsq + corrections** → PPL ~59 (49K correction params, total ~1.6M)
2. **topk+lstsq+polish** → PPL ~70.7 (1.58M params)

### Dead Ends (this session)
Attention pruning, cross-block sharing, exact pruning, GELU affine maps, G-weighted metrics, method hybrids, eigenstructure selection, progressive stacking, corrections on CE-trained models, higher-rank joint, joint compress+correct, warm-start joint, correction-first compression, block-adaptive ratio, centroid-affinity, GELU dead channel census, per-block error concentration, wider-shallower compression, lstsq target alternatives (norm-weighted/SVD-subspace/freq-weighted)

## Next (agenda refreshed — 10 items, 174-183)
- **174-179**: GELU interpretability suite (P0) — positive channel ablation, weight-data alignment, constant bias, sign patterns, training dynamics, bias shift under compression
- **180**: Literature scan: FFN compression 2025-2026 (P1)
- **181**: Wider-shallower compression: 12 blocks at k=8 vs 4 at k=4 (P1)
- **183**: Sign-pattern-conditioned piecewise lstsq (P1)
