# Item
20 — Single-affine collapse quality — toy ReLU

# Context
- Cycle 009 refuted GPT-2 single-affine hypothesis: all 12 blocks have L < 0.6, most negative L (mean Jacobian deviation >> mean Jacobian norm). No GPT-2 block is near-linear; each token activates genuinely different local maps.
- Item 20 asks the toy analog: does at least ONE hidden layer in a trained spirals ReLU network achieve r < 0.15 normalized residual? If yes, that layer is functionally linear on the data and can be collapsed analytically to a single affine map A, b.
- This gates item 21 (GPT-2 FFN single-affine swap). If item 20 is refuted, item 21 is skipped.
- Item depends on: none
- Type: empirical

# Dependencies
- **Hard dependency on trained spirals ReLU network.** Existing code: `aleph/relu_geometric_compression.py` (RegionAtlas, build_region_atlas). Test file: `tests/test_relu_geometric_compression.py`.
- Spirals config must match prior work: 2D input, 3 turns, 2× scale, 4 layers, 64 hidden, trained to >92% accuracy.
- **Probe data required.** Split into fit (first 50% of train) + held-out (second 50% of train).

# Key Files
- `aleph/relu_geometric_compression.py` — region atlas and affine map extraction via MASO formula
- `aleph/datasets.py` — spirals dataset, EmbeddedClassificationBenchmark
- `tests/test_relu_geometric_compression.py` — test utilities for toy networks
- Predecessors: `experiments/high_d_relu_compression.py`, `experiments/rough_spirals_*.py` (reference configs)

# Constraints
- Must measure normalized residual r = ||h_ℓ − Ah_{ℓ−1} − b||_F / ||h_ℓ − mean(h_ℓ)||_F on held-out data (NOT fit data).
- A, b = argmin_A ||h_ℓ(x) − A·h_{ℓ−1}(x) − b||_F via least squares on fit probe data.
- Report r per layer. Confirmed if ≥1 layer achieves r < 0.15. Refuted if all layers r > 0.5.
- Criterion is strict: r quantifies approximation quality, not just linear relationship. A layer with r < 0.15 is genuinely collapsible.

# Handoff
**Setup:** Train spirals ReLU (2→64→64→64→64→1, lr=0.001, 2× scale, 3 turns). Extract region atlas on first 50% of train data. Fit affine maps A_ℓ, b_ℓ per layer via least squares. Measure residual r_ℓ on second 50% (held-out). Report r per layer, confirm/refute based on r < 0.15 threshold for any layer. Confirm hypothesis is only if single-affine collapse works, directly contradicting cycle 009 GPT-2 result (which expected negative outcome and got it).
