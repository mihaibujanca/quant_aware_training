# Research Status — 152 Cycles (001–152)

## Score (estimated)
| Outcome | Count |
|---------|-------|
| Confirmed | 34 |
| Refuted | 52 |
| Inconclusive | 62 |
| Crash | 2 |

## Session 125-152 Summary (28 cycles)
### Confirmed
- FFN isotropy learned (129), curved inference (135), AE param-matched (137)
- Corrections are cascade compensation (151) — novel subspace 93-97%

### Key Findings
- **Joint stacking training** (149): random-init rank-64 beats topk (69.76 vs 70.70, 1.3%)
- **Training objective > architecture** (148): linear = ReLU AE; CE loss is the ingredient
- **Composition barrier** (138): independent training fails, joint training succeeds
- **Progressive stacking fails** (152): 78.2, worse than all baselines

### Working Recipe
1. **Best stacking**: joint random rank-64 + 2000 CE steps → PPL 69.76 (986K params)
2. **Classic**: topk+lstsq+500 polish → PPL 70.70 (1.58M params)
3. **+ LoRA corrections** → PPL ~59

## Next Up
- **163**: Joint training + LoRA corrections (P2, dep on 159)
- **164**: Joint training step budget sweep (P2, dep on 159)
