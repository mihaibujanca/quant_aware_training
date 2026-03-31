# Research Status — 153 Cycles (001–153)

## Session 125-153 (29 cycles)

### Key Findings
1. **Training objective > architecture** (148): linear = ReLU AE. CE loss is the ingredient.
2. **Joint stacking works** (149): random rank-64 = 69.76 (beats topk 70.70, 1.3%, seed-dependent)
3. **CE models resist corrections** (153): joint+corrections = 107 (catastrophic); topk+corrections = 69.8
4. **Corrections are cascade compensation** (151): 93-97% novel subspace, block 7 is entirely cascade
5. **Progressive stacking fails** (152): 78.2, worse than all baselines
6. **Topk+lstsq+corrections remains best overall**: PPL ~59 at 1.6M + 49K params

### Working Recipes (ranked)
1. **topk+lstsq + corrections** → PPL ~59 (49K correction params, total ~1.6M)
2. **Joint random rank-64** → PPL 69.76 (986K params, no corrections needed, but seed-dependent)
3. **topk+lstsq+polish** → PPL ~70.7 (1.58M params)

### Dead Ends (this session)
Attention pruning, cross-block sharing, exact pruning, GELU affine maps, G-weighted metrics, method hybrids, eigenstructure selection, progressive stacking, corrections on CE-trained models

## Next
- **165**: Multi-seed validation of joint training (P0) — 8 seeds to characterize seed-dependence
- **166**: Joint training with correction-aware loss (P0) — train compression + corrections simultaneously
- **167**: Higher-rank joint training rank-128/256 (P1) — test if capacity reduces seed variance
- **168**: Topk+lstsq init for joint training (P1) — warm start from known-good solution
- **169**: Correction-first compression (P2) — flip pipeline: correct then compress residual
