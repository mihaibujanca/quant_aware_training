# Cycle 013 Handoff: RTG Spectral Coarsening (Item 24)

## Item
**RTG spectral coarsening: Fiedler-based region merging — toy ReLU**
- Build Hamming-1 adjacency graph of observed activation patterns
- Compute Laplacian Fiedler vector, run K-way spectral clustering (K=2,4,8,16)
- Merge patterns in same spectral cluster via weighted-average affine maps
- Measure accuracy recovery vs. random merging baseline
- **Confirm if**: ≥3× region reduction with <5% accuracy drop, outperforms random by ≥5pp
- **Refute if**: spectral merging no better than random
- **Inconclusive if**: benefit is geometry-dependent (2D but not 100D)

## Context
- **Cycle 012 refuted item 22** (k-means on raw MASO affine maps): K*/N=1.0, all 375 patterns are functionally distinct in Euclidean affine-map space
- k-means failure **does NOT mean regions are non-redundant** — just that Euclidean distance is the wrong metric
- **Graph-based merging** (item 24) is the natural next step: respects Hamming-1 adjacency structure that k-means ignored
- **Motivation**: Bosca 2025, Dhayalkar 2025 suggest RTG spectral structure carries real merging signals

## Dependencies
- None. Can run immediately.

## Key Files
- `aleph/relu_geometric_compression.py` — extraction, merging, accuracy evaluation
- Test network: spirals 2→32→32→1, trained to 98.3% accuracy (cycle 012 artifact reusable)
- Existing cycle 012 artifacts: `autoresearch/results/cycle_012_kmeans_toy.json`, `cycle_012_summary.md`

## Constraints
- Use weighted average of affine maps within cluster (weight by region support size)
- Include random merging baseline for K=2,4,8,16 (null hypothesis)
- Test on both 2D and 100D-embedded spirals to rule out geometry-specificity
- Report: accuracy curves, spectral vs. random comparison, Fiedler vector visualization

## Handoff
Run item 24 experiment (RTG spectral coarsening on toy ReLU). If spectral clustering outperforms random, proceed to item 25 (GPT-2 analog). If spectral fails, test item 34 (region volume Pareto) as alternative compression signal.
