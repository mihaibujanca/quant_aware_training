# Architecture

## Overview

Quantization-aware training research repo. Studies the geometry of quantization
error through neural network layers.

## Package: `aleph/`

- `quantization.py` — quantization primitives (fake-quant, rounding modes)
- `models.py` — model definitions
- `datasets.py` — dataset loading utilities
- `visualization.py` — plotting helpers

### `aleph/qgeom/` — quantization geometry library

- `core.py` — core quantization geometry operations
- `geometry_2d.py`, `geometry_3d.py` — 2D/3D geometry primitives
- `drawing_2d.py`, `drawing_3d.py` — visualization for geometry
- `manifolds.py` — manifold generation and analysis
- `experiment.py` — experiment scaffolding
- `metrics.py` — shared geometry/entropy/fate metrics and layer report builder
- `policy.py` — geometry-guided correction-point scoring and policy simulation helpers
- `transformer_analysis.py` — transformer-specific instrumentation that emits run geometry reports

## Experiments

- `experiments/` — experiment configs and results
- `run_experiment.py` — main experiment runner (Hydra config)
- `configs/` — Hydra config files
- `analyze_sweeps.py` — sweep result analysis

## Notebooks

- `notebooks/` — exploratory analysis (see [NOTEBOOKS.md](../.context/NOTEBOOKS.md))
