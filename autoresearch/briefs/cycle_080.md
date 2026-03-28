# Cycle 080: Retained-spectrum denoising via analytic projection — GPT-2 FFN

## Item
Item 92: Cycle 072 showed stacking error lives in the retained spectrum (77-91%). Gradient polish effectively denoises this. Test whether a single-step analytic projection can replace 500-step Adam polish: after topk+lstsq, compute per-block error, project onto retained SVD basis, re-solve lstsq on corrected targets. Compare vs Adam polish on stacked PPL for blocks [7,5,4,1] at 6x.

## Context
- Cycle 072 (item 83, REFUTED original hypothesis but CONFIRMED retained-spectrum dominance): 77-91% of error in retained directions, stacking amplifies 1.8-2.6x without shifting profile.
- Cycle 061: 500 Adam steps reduce stacking PPL from +30.2% to +4.9% (baseline ~57.65).
- Cycle 077: basis choice irrelevant (random ~ SVD at k/n=0.67). But the question here is about denoising the retained component, not basis selection.

## Dependencies
- Item 83 (cycle 072, complete): Retained-spectrum error fractions and norms. Result: `autoresearch/results/cycle_072_stacking_error_spectrum.json`.
- Cycle 061 (complete): Adam polish baseline. Result: `autoresearch/results/cycle_061_gradient_polish_scaling.json`. 500-step Adam → +4.9% PPL.

## Key Files
- `experiments/gradient_polish_stacked_gpt2.py` — Adam polish reference (cycle 059/061)
- `experiments/stacking_error_spectral_decomposition.py` — spectral decomposition (cycle 072)
- Output: `autoresearch/results/cycle_080_analytic_projection.json`, `cycle_080_analytic_projection.png`

## Constraints
- Confirmed if analytic correction achieves <= +10% PPL (within 2x of Adam's +4.9%). Refuted if > +25% or diverges. Inconclusive if +10-25%.
- 3 seeds [42, 123, 7]. Blocks [7,5,4,1] at 6x. 4096 probe tokens.
