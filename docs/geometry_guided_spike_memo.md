# Geometry-Guided Quantization Spike Memo

## Purpose

This memo captures the current 2-week spike implementation for geometry-guided quantization correction and records what transfers from low-dimensional intuition to a high-dimensional transformer anchor.

## What Was Added

- Shared report schema for both low/high-dimensional analysis:
  - `LayerGeometryReport`
  - `RunGeometryReport`
- Shared metric toolbox:
  - `compute_geometry_metrics(...)`
  - `compute_entropy_metrics(...)`
  - `compute_fate_metrics(...)`
  - `build_layer_geometry_report(...)`
- Policy toolbox:
  - `score_correction_points(...)`
  - `build_baseline_policy(...)`
  - `simulate_policy(...)`
- Transformer instrumentation:
  - `collect_transformer_layer_reports(...)`
- Notebook-first deliverables:
  - `notebooks/lowdim_geometry_report_card.py`
  - `notebooks/transformer_geometry_report_card.py`
  - `notebooks/policy_comparison_report_card.py`

## Transfer Principles (Low-Dim -> High-Dim)

1. **Linear vs nonlinear error split remains useful.**
   Layers/quant points with high linear fraction and low collapse indicators are the best correction targets.
2. **Fate metrics remain predictive.**
   Flip/collapse behavior tracks where corrections stop working reliably.
3. **Entropy + anisotropy metrics add ranking signal.**
   They help break ties between similar linear-error layers and prioritize stable correction points.

## Current Recommendation

Use geometry-guided placement as the default ranking policy, and keep evenly-spaced placement as the control baseline in all future correction sweeps.

## Next Actions

1. Expand policy comparison to 2-3 seeds.
2. Add Spearman correlation checks between predicted gain and observed gain.
3. Add one MNIST transfer sanity run (classification or reconstruction) using identical report schema.
