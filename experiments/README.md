# Experiments

Active experiment scripts.

| File | Description |
|------|-------------|
| `lambda_sweep.py` | Main sweep for hybrid distillation. Sweeps lambda (layer loss weight), bits, correction_hidden, and seeds across classification/autoencoder/transformer tasks. 162 runs per task. |
| `layer_distillation.py` | Hybrid distillation: output MSE + per-layer activation MSE. Tests whether layer-level supervision helps correction networks learn. |
| `output_distillation.py` | Output-only distillation baseline. Correction networks trained to match float model final outputs only. |
| `quantization_geometry.py` | Geometric analysis of quantization error regions. Tracks how hypercube error regions evolve through ReLU and linear layers. |
| `geometry_overnight.py` | Overnight runner for Q1/Q2 geometry hypotheses: linear recoverability before collapse, and generic corrector ablations using quantization-time statistics. Writes verdict-ready summaries. |
