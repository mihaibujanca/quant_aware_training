# Research Status — 158 Cycles (001–158)

## Session 125-158 (34 cycles)

### Key Findings
1. **Training objective > architecture** (148): linear = ReLU AE. CE loss is the ingredient.
2. **Joint training exhausted** (155-158): random-init 91.1, warm-start 73.0, polish 70.7 — 5 consecutive refutations
3. **Higher rank makes joint training worse** (157): r128=175, r256=622 vs r64=91
4. **CE models resist corrections** (153, 156): joint+corrections=107-110; topk+corrections=59
5. **Corrections are cascade compensation** (151): 93-97% novel subspace
6. **Topk+lstsq+corrections remains best overall**: PPL ~59 at 1.6M + 49K params

### Working Recipes (ranked)
1. **topk+lstsq + corrections** → PPL ~59 (49K correction params, total ~1.6M)
2. **topk+lstsq+polish** → PPL ~70.7 (1.58M params)
3. **topk+lstsq warm-start joint** → PPL ~73.0 (986K params, reliable but worse than polish)

### Dead Ends (this session)
Attention pruning, cross-block sharing, exact pruning, GELU affine maps, G-weighted metrics, method hybrids, eigenstructure selection, progressive stacking, corrections on CE-trained models, higher-rank joint, joint compress+correct, warm-start joint

## Next
- **169**: Correction-first compression (P2) — flip pipeline: correct then compress residual
- After 169: NEXT will be empty → agenda refresh required
