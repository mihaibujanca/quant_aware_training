# Item
**103: Low-rank Jacobian factorization for FFN compression — GPT-2 FFN**

# Context
- Cycle 085 audit (GELU pattern diversity) refuted piecewise-constant compression via clustering (patterns near-unique, ~2045/2048 binary unique states).
- Audit rec: pivot to low-rank Jacobian approximation instead of piecewise-constant.
- Status.md: compression recipe maxes at k≤4 blocks (topk+lstsq+500 Adam), +56% single-block PPL. Joint FFN+attn super-additive. Cascade c≈0.41/block.
- Cycle 099: error is diffuse (rank_95≈500–636/768, ~85% of residual dims). Projection corrections ruled out.

# Dependencies
- No direct blockers.
- Related prior work: cycles 068–071 explored Jacobian rank census, SVD basis selection, subspace alignment.
- Cycle 088: input-space Jacobian clustering showed rank varies with input (max rank 500–650). This item exploits token-wise rank variation.

# Key Files
- `experiments/ffn_jacobian_svd_compression.py` — cycle 071 (SVD vs channel selection baseline).
- `experiments/ffn_jacobian_rank_census.py` — cycle 068 rank census code (reusable).
- `experiments/input_space_jacobian_clustering_gpt2.py` — cycle 088 (per-token Jacobian capture).

# Constraints
- Confirmed if: rank-64 factorization ≤ +15% PPL for ≥2/4 blocks.
- Refuted if: per-token effective rank > 200 OR rank-64 > +30% PPL everywhere.
- Inconclusive: rank varies wildly across blocks/tokens.
- Single-block swap baseline: ~+56% at k=4 topk+lstsq (cycle 098).

# Handoff
Test per-token J(x) (768×768) low-rank structure: compute SVD on 2048 tokens for blocks [1,4,7,11]. Fit global U,V via stacked SVD. Predict S(x) via small net (768→r→r). At r=64: ~100K params vs 4.7M. Measure single-block swap PPL vs topk baseline. Low-rank ansatz is alternative to piecewise-constant (cycle 085 failed) — verifies whether Jacobian structure (not pattern clustering) enables compression.
