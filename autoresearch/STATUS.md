# Research Status — 176 Cycles (001–176)

## Session 125-176 (52 cycles)

### Key Findings
1. **GELU negative regime = constant bias** (173, 174): absorbable in all 12/12 blocks (max +1.4% PPL); token-dependent signal is random noise (cosine ≈ 0.03, matching random baseline)
2. **z negativity is learned** (165, 168): trained Wx_mean=-0.6 to -1.1 vs random≈0; scales with model size (small -0.97, large -1.43)
3. **Positive regime is critical** (164): killing positive channels is 380x more destructive than killing negative
4. **Literature scan found 2 untried methods** (170): SoLA split, PruneNet spectral selection
5. **Topk near-optimal** (161, 141, 143): centroid-FPS, importance pruning, eigenstructure all ≤topk
6. **Block count is primary cost** (171): 12@3x=183 PPL vs 4@6x=59 PPL; wider-shallower catastrophically worse
7. **Lstsq, sign-patterns, piecewise fitting all dead ends** (172, 176): target alternatives within noise; sign-pattern clustering causes overfitting
8. **Topk+lstsq+corrections remains best**: PPL ~59 at 1.6M + 49K params

### Working Recipes (ranked)
1. **topk+lstsq + corrections** → PPL ~59 (49K correction params, total ~1.6M)
2. **topk+lstsq+polish** → PPL ~70.7 (1.58M params)

## Next
- 4 items queued (187-190): SoLA split (P0), negative-regime absorption (P0), PruneNet spectral (P1), FLAP bias compensation (P1)
- Priority directions: split compression (preserve+compress vs prune+reconstruct), reduce effective problem size via bias absorption
