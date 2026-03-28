# Item
Item 81: Shared vs token-specific Jacobian spectral structure — GPT-2 FFN

# Context
Item 80 (cycle 068) found that 10/12 blocks have low effective rank (rank_99 ≤ 500), but rank does NOT predict compression difficulty (rho=−0.33, p=0.30). Item 81 validates whether this low-rank structure is globally shared across tokens or token-specific.

# Dependencies
- Item 80 COMPLETE: cycle 068 results in `autoresearch/results/cycle_068_jacobian_rank_census.json` — Jacobian rank_99 per block, 3 seeds×516 tokens, random baseline 592
- No code dependencies; method is standalone analysis

# Key Files
- `autoresearch/results/cycle_068_jacobian_rank_census.json` — rank stats per block (12 blocks, 3 seeds, 516 tokens ea)
- No existing code; use numpy/torch SVD on Jacobians over 256 probe tokens per block
- Output: `autoresearch/results/cycle_069_jacobian_subspace_alignment.json` with mean alignment cosine, variance, per-block breakdown

# Constraints
- Use k from item 80 effective rank OR k=384 (8x compression baseline)
- Measure pairwise principal angles (subspace alignment) across tokens — use smallest principal angle cosine as alignment metric
- 256 tokens minimum (stated in item text); use all 516 from cycle 068 if available
- Report: mean alignment, variance, per-block results

# Handoff
**Confirm/refute**: Is Jacobian spectral structure globally shared (mean alignment ≥ 0.8 for ≥10/12 blocks)? Or token-specific (< 0.5 for majority)? This determines whether topk channel selection is near-optimal or leaves room for adaptive compression. If high alignment, low-rank claim stands despite rank not predicting damage (cycle 068).
