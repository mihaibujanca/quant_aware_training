# Cycle 016 Plan — GELU PWL Approximation Fidelity

## Item
32. GELU → 3-piece PWL approximation fidelity — toy GELU MLP

## Criteria (verbatim)
- Confirmed if: 3-piece PWL substitution causes < 3% accuracy drop and 5-piece causes < 1%, indicating that GELU networks are approximately piecewise-affine at low piece count.
- Refuted if: even 10-piece PWL causes > 5% accuracy drop, indicating that smooth curvature is task-critical and the piecewise-affine approximation is fundamentally flawed.
- Inconclusive if: accuracy recovers only with a large number of pieces (> 10), suggesting GELU networks are piecewise-affine but at much finer granularity than ReLU.

## Pre-flight
- P1 dependency: none.
- P2 data: spirals (n=2000, 3 turns, scale=2.0, seed=42) via `aleph.datasets`.
- P3 compute: single MLP training + 4 forward-pass sweeps. Under 2 min.
- P4 criteria: quantitative thresholds above — no ambiguity.

## Design

### Network
- Architecture: 2D → 100D fixed random projection → 4 hidden layers × 64 → 1.
- Activation: `torch.nn.GELU()`.
- Train: Adam lr=0.001, BCE loss, up to 5000 epochs, early-stop at acc > 92%.
- Also train a 2D variant (no embedding) as sanity check if 100D is unstable.

### PWL approximation
- Domain: [-3, 3]. Outside: g̃(x)=0 for x<-3, g̃(x)=x for x>3.
- For K pieces: K+1 uniformly spaced breakpoints in [-3, 3].
- Values at breakpoints: exact GELU(x_i).
- Interpolation: `torch.lerp` between adjacent breakpoints (vectorized).
- Sweep K = {1, 3, 5, 10}.

### Evaluation
- Swap GELU → PWL (no retraining), evaluate test accuracy for each K.
- Record: float GELU acc, PWL acc per K, absolute and relative drop.
- Plot: accuracy vs K with horizontal lines at 3% and 1% drop thresholds.

## Output
- Script: `experiments/gelu_pwl_approximation.py`
- Plot: `experiments/gelu_pwl_approximation.png`
- Outcome: confirmed / refuted / inconclusive per criteria.
- If refuted: skip item 33.

## Risks
- 100D embedding + GELU + lr=0.001 may underfit. Fallback: 2D-only variant.
- GELU tails outside [-3,3] may matter. Mitigation: check activation histogram.
