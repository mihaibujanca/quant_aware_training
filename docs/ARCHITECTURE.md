# Architecture

## Overview

Research repo about using geometry to understand and compress neural networks.

The codebase currently contains two substantial branches:

1. Quantization / correction experiments
2. Geometric compression experiments

The current intended direction is geometric compression, but the quantization
stack remains active, tested, and still represents a large part of the package.

## Package: `aleph/`

- `models.py` — shared model definitions for MLP, autoencoder, transformer, and correction modules
- `datasets.py` — dataset loading utilities for spirals, MNIST, Shakespeare, and embeddings
- `visualization.py` — plotting helpers for decision boundaries and correction views
- `training.py` — training utilities for correction/QAT workflows
- `quantization.py` — fake-quant, calibration, rounding-mode, and QAT utilities
- `differential.py` — Jacobian / boundary-density helpers for geometric profiling
- `mesh_datasets.py` — mesh loading and SDF sampling for geometric compression
- `geometric_compression.py` — source-MLP to compressed-model pipeline on implicit 3D shapes
- `relu_geometric_compression.py` — data-active region atlas and smaller-ReLU synthesis on embedded low-D classification tasks
- `boundary_geometric_compression.py` — boundary-centric compression on embedded 2D shape-classification tasks; projects the teacher to 2D, simplifies flat boundary zones, and uses spline patches only in high-curvature zones

### `aleph/qgeom/` — quantization geometry library

- `core.py` — core quantization geometry operations
- `geometry_2d.py`, `geometry_3d.py` — 2D/3D geometry primitives
- `drawing_2d.py`, `drawing_3d.py` — visualization for geometry
- `manifolds.py` — manifold generation and analysis
- `experiment.py` — experiment scaffolding
- `metrics.py` — shared geometry/entropy/fate metrics and layer report builder
- `policy.py` — geometry-guided correction-point scoring and policy simulation helpers
- `transformer_analysis.py` — transformer-specific instrumentation that emits run geometry reports

### Current entry points

- [`run_experiment.py`](/Users/mihai/Projects/quant_aware_training/run_experiment.py) — Hydra-based entry point for the quantization/correction branch
- [`scripts/run_shape_sweep.py`](/Users/mihai/Projects/quant_aware_training/scripts/run_shape_sweep.py) — main batch runner for geometric compression
- [`experiments/kan_last_stand_overnight.py`](/Users/mihai/Projects/quant_aware_training/experiments/kan_last_stand_overnight.py) — decisive KAN viability sweep

## Experiments

- `experiments/` — experiment scripts for both branches
- `configs/` — Hydra config files used by `run_experiment.py`
- `scripts/` — operational runners for shape sweeps and speed-tier calibration
- `analyze_sweeps.py` — sweep result analysis for older quantization experiments

### Quantization / correction experiments

- `lambda_sweep.py`
- `layer_distillation.py`
- `output_distillation.py`
- `geometry_overnight.py`
- `quantization_geometry.py`
- `transformer_correction_analysis.py`

### Geometric compression experiments

- `curvature_boundary_compression.py`
- `kan_last_stand_overnight.py`
- `projected_kan_sweep.py`
- `jacobian_annealing_sweep.py`

### Historical archive

- `old_experiments/` — superseded scripts kept for provenance and comparison
- `autoresearch/` — planning, survey, and research-program support material

## Notebooks

- `notebooks/` — exploratory analysis for both research branches
- The notebook layer mixes active exploration, reports, and backups; treat it as research support, not the canonical architecture
