# Cycle 071: Spectral Compression (Item 82)

## Item
Jacobian SVD basis compression vs axis-aligned topk. Blocks [1, 4, 7, 11], 512 tokens.
Compare: topk+lstsq (baseline) vs Jacobian SVD projection + lstsq (test).
Metric: PPL improvement ≥3% in ≥2/4 blocks → Confirmed.

## Context
Cycle 068 (item 80): Measured FFN Jacobian rank_99 per block (433–589).
Cycle 026: Weight SVD lost to topk+lstsq. Now testing *data-conditioned* Jacobian SVD.
Current pipeline: topk channel selection (axis-aligned, suboptimal in theory).

## Dependencies
Item 80 (cycle 068): Complete. Jacobian statistics in `cycle_068_jacobian_rank_census.json`.

## Key Files
- `aleph/lm_geometric_compression.py` — FFN compression pipeline
- `autoresearch/results/cycle_068_jacobian_rank_census.json` — rank stats
- Output: `autoresearch/results/cycle_071_jacobian_svd_basis.json`

## Constraints
- Matched budget: k = hidden_dim / compression_ratio (6x baseline)
- Test variants: (a) SVD-only, (b) SVD+lstsq
- Single-block swap, then report relative PPL delta

## Handoff
Implement Jacobian SVD basis for blocks [1,4,7,11]. Compute mean Jacobian over 512
probe tokens (OpenWebText, seed 42). Project FFN output onto top-k left singular vectors.
Measure PPL vs baseline topk+lstsq. Report per-block deltas.
