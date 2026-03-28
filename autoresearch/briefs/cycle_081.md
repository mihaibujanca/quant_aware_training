# Cycle 081: Analytic stacking correction via Jacobian chain rule — GPT-2 FFN [MOONSHOT]

## Item
Item 85: Gradient polish works (cycle 061: +30% -> +4.9% at 500 Adam steps) but is iterative. Can we compute the same correction analytically via a single Newton step? For each compressed block i, compute the gradient g = dLoss/d(vec(W_down_i)) and approximate Hessian H = J^T J through the full stacked model, then apply Delta_W = -H^{-1} g. This accounts for cross-block error propagation that per-block lstsq (cycle 080) cannot see.

## Context
- Cycle 061: 500 Adam steps on W_down achieves +4.9% PPL on OWT (baseline 57.65, blocks [7,5,4,1] at 6x).
- Cycle 080: Per-block analytic lstsq re-solve is a no-op (+10.6% -> +10.6%). Local correction cannot see stacking error.
- This item is fundamentally different: it uses the full model loss gradient, not per-block MSE. The Newton step linearizes the downstream path.

## Dependencies
- Cycle 061 (complete): Adam polish baseline on OWT. Unpolished +30.2%, 500 steps -> +4.9%.
- Cycle 080 (complete): Analytic projection is vacuous. Motivates global (not local) approach.

## Key Files
- `experiments/gradient_polish_scaling_gpt2.py` — cycle 061 Adam polish (OWT, 500 steps)
- `experiments/gradient_polish_stacked_gpt2.py` — cycle 059 polish framework
- Output: `autoresearch/results/cycle_081_newton_correction.json`, `cycle_081_newton_correction.png`

## Constraints
- Confirmed if single Newton step achieves <= +8% PPL (within 2x of Adam's +4.9%).
- Refuted if Newton step diverges or gives > +20% PPL.
- Inconclusive if Newton step helps (+8-20%) but doesn't match Adam.
- MUST use OWT corpus for comparability with cycle 061.
- 3 seeds [42, 123, 7]. Blocks [7,5,4,1] at 6x.
