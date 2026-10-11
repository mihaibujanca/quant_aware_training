# Experiments

Active experiment scripts.

## Quantization / Correction

| File | Description |
|------|-------------|
| `quant_geometry_foundations.py` | Exact weight-only map/ledger and shared MLP oracle examples. Reproduces eight explanatory figures, including a numerical-rank audit, and a run manifest; see [foundations report](../docs/quant_geometry_foundations.md) and [progress tracker](../docs/quant_geometry_progress.md). |
| `lambda_sweep.py` | Main sweep for hybrid distillation. Sweeps lambda (layer loss weight), bits, correction_hidden, and seeds across classification/autoencoder/transformer tasks. 162 runs per task. |
| `layer_distillation.py` | Hybrid distillation: output MSE + per-layer activation MSE. Tests whether layer-level supervision helps correction networks learn. |
| `output_distillation.py` | Output-only distillation baseline. Correction networks trained to match float model final outputs only. |
| `quantization_geometry.py` | Geometric analysis of quantization error regions. Tracks how hypercube error regions evolve through ReLU and linear layers. |
| `geometry_overnight.py` | Overnight runner for Q1/Q2 geometry hypotheses: linear recoverability before collapse, and generic corrector ablations using quantization-time statistics. Writes verdict-ready summaries. |
| `correction_depth_scaling.py` | Depth/capacity study for correction quality on quantized MLPs. |
| `correction_distillation_architectures.py` | Architecture comparison for correction distillation variants. |
| `correction_distillation_sweep.py` | Sweep runner for correction distillation setups. |
| `canonical_overnight.py` | Overnight run focused on canonical error/correction framing. |
| `transformer_correction_analysis.py` | Transformer-side analysis and reporting utility. |

## Geometric Compression

| File | Description |
|------|-------------|
| `curvature_boundary_compression.py` | Boundary-centric compression runner on embedded 2D shape-classification tasks. Defaults to harder latent-2D benchmarks with localized high-curvature windows and hairpins, then extracts a projected teacher boundary, simplifies flat zones, and allocates spline patches only in high-curvature zones. |
| `high_d_relu_compression.py` | High-D-first ReLU compression runner on embedded low-dimensional classification datasets (`2->100`, `2->1000`, with native `2D` as reference only). |
| `kan_last_stand_overnight.py` | Favorable-budget overnight sweep intended to decide whether low-capacity KAN compression is worth keeping. |
| `projected_kan_sweep.py` | Tests projection-then-KAN compression, where a low-dimensional subspace is extracted first. |
| `jacobian_annealing_sweep.py` | Compares Jacobian matching schedules to test whether softer geometry constraints improve fit quality. |

## Mixed / Geometry Analysis

| File | Description |
|------|-------------|
| `geometric_decomposition_sweep.py` | Decomposes quantization error into metric vs topological distortion across controlled classification settings. |
