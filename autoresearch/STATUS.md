# Research Status

## Completed (last 5)
| Cycle | Item | Type | Outcome | Key Finding |
|-------|------|------|---------|-------------|
| 019 | Entropy cascade — toy | empirical | confirmed | cH drops 96% L1→L3/L4; correlates with affine compressibility |
| 018 | Procrustes linearity — GPT-2 | empirical | inconclusive | 7/12 layers lin≥0.98; block 6 identity +3.9% ppl |
| 017 | Region volume Pareto — toy | empirical | inconclusive | Gini=0.408; top 20% covers only 48% |
| 016 | GELU PWL — toy | empirical | inconclusive | K=10 drops 2.6%, K=3 drops 35pp |
| 015 | Intrinsic dimension — GPT-2 | empirical | inconclusive | ID rises through depth |

## Key Narrative
Two positive signals have emerged:
1. **Procrustes linearity (cycle 018)**: 7/12 GPT-2 layers are near-identity. Block 6 removable at +3.9% ppl. Spearman with damage = 0.56.
2. **Entropy cascade (cycle 019)**: Deep toy layers add zero new partition information. This explains WHY single-affine collapse works on deep layers (cycle 010).

The toy and GPT-2 stories are CONSISTENT: in both architectures, deeper layers do less novel computation. But the effect is much stronger in toy ReLU (96% conditional entropy drop) than in GPT-2 (where even "linear" blocks can't be jointly removed).

## Next
Eligible P0: 38 (manifold curvature GPT-2, dep 9 done), 39 (grokking toy, no deps), 40 (attention head census GPT-2, dep 9 done).

## Dead Ends
- Tropical, stable-channel, single-affine (GPT-2), k-means/spectral on maps, intrinsic dim, coarse PWL, raw volume pruning
