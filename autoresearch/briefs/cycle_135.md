# Item 145: Curved Inference Census — GPT-2 Residual Stream

## Item
Validate the curved inference framework (Manson 2025) on GPT-2. Collect residual trajectory x^(0)..x^(12) over 200 OWT sequences. Compute per-layer salience S_l = ||delta||_G and curvature kappa_l under pullback metric G = U^T U. Compute per-block single-block swap PPL delta at 6x (all 12 blocks, topk+lstsq). Report salience-curvature correlation and curvature-vs-swap-damage correlation.
- **Confirmed**: salience-curvature |rho| >= 0.7 AND curvature-damage |rho| >= 0.5.
- **Refuted**: |salience-curvature rho| < 0.3.
- **Inconclusive**: anticorrelation exists (|rho| >= 0.3) but curvature-damage weak (|rho| < 0.3).

## Context
- Manson reports salience-curvature r=-0.89 on LLaMA-3.2-3b. GPT-2 is 10x smaller (12 layers vs 28), so correlation has only 11 curvature data points — inherently noisy.
- Survey: `autoresearch/surveys/curved_inference_geometric_interpretability.md`. Pullback metric G = U^T U maps residual-space distances to output-probability shifts.
- Library: `aleph/pullback_geometry.py` has `collect_residual_trajectory`, `compute_salience`, `compute_curvature`, `get_unembedding` — all needed functions exist.
- ArXiv 2507.21107 (July 2025) — paper exists and is surveyed, not speculative.
- Item 146 (pullback error measurement) depends on this item validating the metric on GPT-2.

## Dependencies
- None hard. GPT-2 pretrained + OWT eval data.

## Key Files
- `aleph/pullback_geometry.py` — all metric computations
- `autoresearch/surveys/curved_inference_geometric_interpretability.md` — full survey
- Existing swap-PPL pattern: `experiments/placement_sweep_k4_cycle125.py` (topk+lstsq+polish pipeline)
