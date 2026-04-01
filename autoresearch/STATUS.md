# Research Status — 180 Cycles (001–180)

## Session 125-180 (56 cycles this session)

### Key Findings
1. **GELU negative regime = constant bias** (173, 174): absorbable in 12/12 blocks; token-dependent signal is random noise
2. **z negativity is learned, scales with model size** (165, 168): trained Wx_mean=-1 vs random≈0; larger models push more into dead zone
3. **Positive regime is the computation carrier** (164): 380x more destructive to kill than negative
4. **SoLA split: marginal at best** (177): 0.97 PPL gain at 3.57x params — poor tradeoff
5. **Negative absorption impossible** (178): 94-99% of channels are sometimes positive, no channel reduction
6. **Spectral selection worse than topk** (179): +3.78 PPL worse, weight-only spectral preservation misses activation importance
7. **FLAP bias compensation: weak init but polish recovers** (180): bias-only=91.9, bias+polish=52.3 (vs lstsq+polish=49.8)
8. **Block count is primary cascade cost** (171): 12@3x=183 vs 4@6x=59 PPL
9. **Topk+lstsq+corrections remains best**: PPL ~59 at 1.6M + 49K params

### Working Recipes (ranked)
1. **topk+lstsq + corrections** → PPL ~59 (49K correction params, total ~1.6M)
2. **topk+lstsq+polish** → PPL ~70.7 (1.58M params)

### Exhausted Directions
Channel selection (topk optimal), joint training (5 refutations), dead channel pruning (5 refutations), block-adaptive ratio, correction transfer, lstsq targets, sign-pattern partitioning, SoLA split, negative absorption, spectral selection, FLAP bias, wider-shallower, per-block concentration

## Next
- NEXT is empty — agenda refresh required
- The recipe has been stable for 60+ cycles. New items should explore fundamentally different approaches.
