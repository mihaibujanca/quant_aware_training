# Research Status — 159 Cycles (001–159)

## Session 125-159 (35 cycles)

### Key Findings
1. **Training objective > architecture** (148): linear = ReLU AE. CE loss is the ingredient.
2. **Joint training exhausted** (155-158): random-init 91.1, warm-start 73.0, polish 70.7 — 5 consecutive refutations
3. **Correction subspaces are compression-specific** (159): corrections trained on compressed model are OOD for float residuals; transferred corrections worse than none (75.3 vs 80.2)
4. **CE models resist corrections** (153, 156): joint+corrections=107-110; topk+corrections=59
5. **Corrections are cascade compensation** (151): 93-97% novel subspace
6. **Topk+lstsq+corrections remains best overall**: PPL ~59 at 1.6M + 49K params

### Working Recipes (ranked)
1. **topk+lstsq + corrections** → PPL ~59 (49K correction params, total ~1.6M)
2. **topk+lstsq+polish** → PPL ~70.7 (1.58M params)
3. **topk+lstsq warm-start joint** → PPL ~73.0 (986K params, reliable but worse than polish)

### Dead Ends (this session)
Attention pruning, cross-block sharing, exact pruning, GELU affine maps, G-weighted metrics, method hybrids, eigenstructure selection, progressive stacking, corrections on CE-trained models, higher-rank joint, joint compress+correct, warm-start joint, correction-first compression

## Next (agenda refreshed)
- **170**: Block-adaptive compression ratio (P1) — non-uniform k per block
- **171**: Centroid-affinity channel selection (P1) — geometry-based channel selection vs topk
- **172**: GELU-aware dead/saturated channel census (P2) — R2 exact white-box baseline
- **173**: Per-block error contribution profiling (P2) — identify cascade error sources
