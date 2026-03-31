# Research Status — 148 Cycles (001–148)

## Score
| Outcome | Count |
|---------|-------|
| Confirmed | 32 |
| Refuted | 50 |
| Inconclusive | 60 |
| Crash | 2 |

## Session 125-148 Results
- **Confirmed**: isotropy learned (129), curved inference (135), AE param-matched (137)
- **Refuted**: placement (125), Procrustes (130), attention pruning (132), hybrid (133), G-error (136), AE stacking (138), eigenstructure (143), layer-local pullback (144), linear AE ablation (148)
- **PLAN INVALID**: 126, 139, 140, 145, 147
- **Inconclusive**: corrections novel subspace (127), dead neurons (128), AE POC (131), affine maps (134), importance pruning (141), correction subspace (142), nanochat audit (146)

## Key Findings (cycles 143-148)
- **Training objective > architecture** (cycle 148): Linear AE = ReLU AE within 0.7%. CE loss training is the active ingredient, not encoder nonlinearity.
- **Composition barrier persists** (cycle 138): AE beats topk 3% locally, loses 4.7% stacked. Independently trained blocks don't compose.
- **No-training selection near-optimal** (cycles 141, 143): importance pruning (+0.2-2%), eigenstructure whitening (WORSE) — topk is robust.
- **Frobenius is best error metric** (cycle 144): Layer-local pullback (rho=0.538) worse than simple Frobenius (rho=0.797).

## Working Recipe (unchanged)
1. FFN k≤4: topk + lstsq + 500 Adam at 6× → PPL ~70
2. + LoRA corrections (r=8, 49K params): PPL ~59

## The Fundamental Barrier
**Composition > Local Quality.** AE beats topk 3% locally but loses 4.7% stacked. The untested hypothesis: joint training under stacked CE loss may address the composition barrier directly.

## Next Up
- **159**: Joint stacking-aware compression training (P0, no deps)
- **160**: Linear factorization + CE training, single-block (P0, no deps)
- **161**: Correction subspace redo on unpolished model (P1, fixes cycle 142)
- **162**: Progressive stacking — train-then-freeze (P1, no deps)
- **163**: Joint training + LoRA corrections (P2, dep on 159)
- **164**: Joint training step budget sweep (P2, dep on 159)
