# Task
Item 81: Shared vs token-specific Jacobian spectral structure — GPT-2 FFN. Measure pairwise subspace alignment of top-k right singular vectors of J(x) across 256 tokens per block.

# Hypothesis
Low-rank Jacobian structure (cycle 068) is globally shared across tokens (mean alignment >= 0.8 for >= 10/12 blocks), supporting global channel selection.

# Steps
1. Create `experiments/ffn_jacobian_subspace_alignment.py`. Load GPT-2, extract W_in/W_out per block (Conv1D `.T`), build probe corpus from shakespeare.txt, compute J(x) = W_out @ diag(GELU'(z)) @ W_in per token.
2. Per token: truncated SVD of J(x) (768x768), keep top-k right singular vectors Vh[:k, :]. Run TWO k settings per block: k_99 = round(mean_rank_99) and k_95 = round(mean_rank_95), both from `autoresearch/results/cycle_068_jacobian_rank_census.json` field `block_stats[str(b)]["mean_rank_99"]` and `["mean_rank_95"]`.
3. Per block per k-setting: sample 1000 random token pairs from the 256 tokens. Alignment metric: mean of svdvals(Vh_i @ Vh_j.T).
4. Random baseline: per block per k-setting, generate 200 pairs of random k-dim subspaces in R^768 (QR of randn(768,k)), compute same metric. Report random_mean_cos and random_std.
5. Compute normalized excess: `excess = (actual - random) / (1 - random)`. This is the PRIMARY interpretive metric. Raw mean_cos is secondary (retained for agenda compatibility).
6. Seeds: [42, 123, 7]. N_TOKENS=256, SEQ_LEN=128, N_SEQUENCES=2. Report cross-seed mean and std.
7. Verdict logic (two layers):
   - Agenda verdict: Confirmed if raw mean_cos >= 0.8 for >= 10/12 blocks (at either k setting). Refuted if < 0.5 for majority. Inconclusive otherwise.
   - Geometric verdict: For each block passing raw >= 0.8, check normalized excess. If excess < 0.1, flag as "trivially confirmed (dimensionality artifact)". Report count of genuinely-confirmed blocks (excess >= 0.1) alongside raw count.
   - Final outcome string: if all confirmed blocks are trivial, outcome = "trivially_confirmed". If >= 10/12 genuinely confirmed, "confirmed". Mix = "partially_confirmed_partially_trivial".
8. Save JSON to `autoresearch/results/cycle_069_jacobian_subspace_alignment.json` with per-block entries containing: k_99, k_95, actual_mean_cos, random_mean_cos, excess, is_trivial (bool), for both k settings.
9. Save bar plot to `autoresearch/results/cycle_069_jacobian_subspace_alignment.png`: grouped bars (actual vs random) per block, one panel for k_95, one for k_99. Horizontal line at 0.8. Annotate trivial blocks.
10. Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/ffn_jacobian_subspace_alignment.py`
11. Update agenda: move item 81 to PAST EXPERIMENTS with outcome. Git commit.

# Validation
- Confirmed if: mean subspace alignment >= 0.8 for >= 10/12 blocks (agenda criterion).
- Refuted if: mean alignment < 0.5 for majority of blocks.
- Inconclusive if: moderate (0.5-0.8) or high variance within blocks.
- Interpretation caveat: right singular vectors measure shared input-space directions that J amplifies, not shared activation patterns directly. This is a valid proxy for whether global low-rank approximation captures per-token Jacobians, but does not test channel selection in activation space.
- If raw alignment >= 0.8 but normalized excess < 0.1, report "trivially confirmed" — high alignment is a geometric inevitability of k/d ratio, not evidence of shared structure.

# Artifacts
- `experiments/ffn_jacobian_subspace_alignment.py`
- `autoresearch/results/cycle_069_jacobian_subspace_alignment.json`
- `autoresearch/results/cycle_069_jacobian_subspace_alignment.png`

# Fallbacks
- If full SVD too slow: use `torch.svd_lowrank(J, q=k+10)` for randomized truncated SVD.
- If memory issue: process one block at a time, free Vh matrices between blocks.
