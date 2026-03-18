# Cycle 017 Brief — Item 34: Region Volume Pareto (Toy ReLU)

## Item
Region volume Pareto: do large regions carry most task accuracy? — toy ReLU.

## Context
Each distinct activation pattern in a ReLU network defines a linear region with its own affine map A_ω.
"Volume" = fraction of test inputs in that region. If a few large regions carry most correctly-classified
points, compression can keep only those regions and route others to their nearest large-region neighbor.

## Dependencies
None (item depends on: none).

## Key Files
- `aleph/datasets.py` — `make_spirals`
- `aleph/relu_geometric_compression.py` — reference for pattern extraction + A_ω formula
- `experiments/high_d_relu_compression.py` — reference for spirals training setup
- Cycle 012 artifacts: `autoresearch/results/cycle_012_kmeans_toy.json`

## Constraints
- Network: 2→32→32→1, seed=42, spirals n=2000, 3 turns, scale=2.0 (cycle 012 setup)
- Test set: 1000 points
- A_ω formula: A_ω = W_L · diag(q^L) · … · W_1 · diag(q^1)
- Fallback: nearest kept region by Hamming distance on activation pattern vectors
- Confirmed if: top 20% of regions by volume covers ≥70% correctly classified test points
- Refuted if: Gini coefficient of volume distribution < 0.3

## Handoff
Experiment script: `experiments/region_volume_pareto.py`
Results: `autoresearch/results/cycle_017_*.{json,png}`
Manifest: `autoresearch/results/cycle_017_manifest.txt`
