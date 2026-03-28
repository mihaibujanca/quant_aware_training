# Cycle 075: Region Volume Pareto — GPT-2 FFN Analog

## Item
Item 35: K-means on per-token FFN Jacobians as proxy for linear regions. Test Pareto structure in cluster volume distribution.

## Context
- Item 34 (cycle 017, toy ReLU): Gini=0.408, top 20% covers 48.2% — partial Pareto, INCONCLUSIVE.
- Item 9 (cycle 009, Jacobian variance atlas): refuted (Spearman=-0.14). This experiment uses per-token Jacobians differently (clustering, not variance).
- Cycle 068/071/072: per-token Jacobian J(x) = W_out @ diag(GELU'(z)) @ W_in infrastructure exists.
- Compression difficulty ground truth: damage_8x/16x from cycle 048 (12 blocks).

## Dependencies
- Item 9: refuted but irrelevant (different use of Jacobians).
- Item 34: INCONCLUSIVE on toy — GPT-2 provides 12 blocks and richer structure.

## Key Files
- `experiments/ffn_jacobian_svd_compression.py` — reuse `_mean_jacobian_svd`, `_gelu_grad_autograd`, `_standard_weight`
- `experiments/region_volume_pareto.py` — cycle 017 reference (toy Pareto analysis)
- `autoresearch/results/cycle_048_local_complexity_gpt2.json` — damage_8x/16x ground truth

## Constraints
- Per-token Jacobians are 768x768. K-means on 589K-dim vectors is infeasible; use SVD singular value profile (768D) or PCA reduction.
- Must include null baseline (uniform cluster assignment) for Gini/coverage comparison.
- 3 seeds minimum for probe corpus variation.
