# Research Status — 142 Cycles (001–142)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 26 |
| Refuted | 41 |
| Inconclusive | 65 |
| Crash | 2 |

## Session 125-142 Results
- **Confirmed**: isotropy learned (129), curved inference (135), AE param-matched (137)
- **Refuted**: placement (125), Procrustes (130), attention pruning (132), hybrid (133), G-error (136), AE stacking (138)
- **PLAN INVALID**: 126, 139, 140
- **Inconclusive**: corrections novel subspace (127), dead neurons (128), AE POC (131), affine maps (134), importance pruning (141), correction subspace (142)

## Working Recipe (unchanged)
1. FFN k≤4: topk + lstsq + 500 Adam at 6× → PPL ~70
2. + LoRA corrections (r=8, 49K params): PPL ~59

## The Fundamental Barrier
**Composition > Local Quality.** AE beats topk 3% locally but loses 4.7% stacked.

## Next Up
- **155**: Eigenstructure-aware channel selection (P1, no deps)
- **149-150**: Layer-local pullback, curvature-guided corrections (P2)
- **156**: Nanochat asset audit (P2)
- **157-158**: Stacked AE variants (P2, dep on 152 which is refuted)
