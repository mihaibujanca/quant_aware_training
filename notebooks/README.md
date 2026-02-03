# Notebooks

Exploratory notebooks. Plots output to `plots/`.

Shared utility functions live in `qgeom/` (geometry primitives, drawing, manifold generation).

| File | Description |
|------|-------------|
| `weight_quantization_geometry.ipynb` | Weight-only quantization error geometry in 2D and 3D. Four 2D experiments (uniform diagonal, non-uniform diagonal, full matrices, circle manifold), 3D extensions (SVD, bounding box efficiency, channel sensitivity), and manifold analysis (error is a linear transform of input). |
| `activation_quantization_geometry.ipynb` | Full fake quantization (weight + activation). Per-tensor activation quantization, weight-only vs full comparison, layer-by-layer error evolution, bit-width comparison (2/4/8-bit). |
| `error_tracking_viz.ipynb` | Visualizes error decomposition through network layers: rounding error, ReLU flip error, and accumulation patterns. |
| `geometric_entropy.ipynb` | Explores the relationship between quantization error volume, entropy, and geometry. Studies how error shape changes through linear vs nonlinear layers. |
| `lowdim_geometry_report_card.ipynb` | Unified geometry/entropy/fate report cards on controlled 2D scenarios (linear-only, ReLU-boundary, saturation-heavy). Includes policy ranking preview. |
| `transformer_geometry_report_card.ipynb` | High-dimensional 4-bit transformer report card using the same metric schema as low-dimensional experiments. |
| `policy_comparison_report_card.ipynb` | Compares geometry-guided correction placement against evenly-spaced baseline under equal correction budget. |
