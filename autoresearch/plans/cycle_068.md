# Task
FFN Jacobian effective rank census on GPT-2. For each of the 12 blocks, compute per-token Jacobian J(x) = W_out.T @ diag(GELU'(z)) @ W_in.T for N=512 OpenWebText probe tokens across 3 seeds. SVD each J(x), report effective rank at 95%/99% energy. Correlate with cycle 048 P1 Frobenius std and cycle 005 single-block damage.

# Hypothesis
- **Confirmed if**: mean effective rank (99% energy) is ≤ 500 for ≥ 10/12 blocks AND Spearman(rank, P1_frob_std) ≥ 0.5.
- **Refuted if**: effective rank is > 700 for a majority of blocks (FFN is not meaningfully low-rank, topk compression is not explained by rank structure).
- **Inconclusive if**: rank is low but does not correlate with P1 or compression difficulty.

# Pre-flight Results
- P1–P4 PASS (see brief). P5: random 768x768 matrix has rank ~768 at 99%; measured ranks well below are meaningful.
- P6: GPT-2 uses Conv1D — `c_fc.weight.shape = (768, 3072)`, `c_proj.weight.shape = (3072, 768)`. Must transpose both: `W_in = c_fc.weight.T` (3072, 768), `W_out = c_proj.weight.T` (768, 3072). Assert shapes before forming J.
- P7: 512 tokens x 3 seeds = 1536 Jacobians per block. Sufficient for mean rank estimation.

# Steps
1. **Script**: Create `experiments/ffn_jacobian_rank_census.py`. Load GPT-2 via transformers. For each seed in [42, 123, 7]:
   - Sample 512 probe tokens from OpenWebText via `aleph/lm_geometric_compression.py` ProbeCorpus.
   - For each of 12 FFN blocks:
     - Extract `W_in = block.mlp.c_fc.weight.T` and `W_out = block.mlp.c_proj.weight.T`. **Assert** `W_in.shape == (3072, 768)` and `W_out.shape == (768, 3072)`.
     - Hook the FFN to capture pre-activation `z = W_in @ x + b` (before GELU).
     - Compute GELU derivative via autograd: `z.requires_grad_(True); g = F.gelu(z); sigma_prime = torch.autograd.grad(g.sum(), z)[0]`.
     - Form `J = W_out @ torch.diag(sigma_prime) @ W_in`, shape `(768, 768)`.
     - SVD: `s = torch.linalg.svdvals(J)`. Energy: `cumsum(s**2) / sum(s**2)`. Record rank_95, rank_99.
   - Aggregate: mean, std, min, max of rank per block per seed.
2. **Control**: Compute rank_99 for 10 random 768x768 matrices; assert all are >=750.
3. **Correlation**: Load `autoresearch/results/cycle_048_local_complexity_gpt2.json`. Extract `p1_frob_std` per layer and `damage_8x`. Compute `spearmanr(mean_rank_99, p1_frob_std)` and `spearmanr(mean_rank_99, damage_8x)`. Correlation direction is **exploratory** (do not pre-specify sign).
4. **Plot**: Bar chart of mean rank_99 per block with cross-seed std error bars. Secondary axis overlays P1 frob_std. Save to `autoresearch/results/cycle_068_jacobian_rank_census.png`.
5. **Save JSON**: `autoresearch/results/cycle_068_jacobian_rank_census.json` — per-block per-seed stats, correlations (rho, p-value), outcome.
6. **Classify** outcome per hypothesis criteria.
7. **Update agenda**: Move item 80 from NEXT to PAST EXPERIMENTS with cycle 068, commit hash, artifact paths.
8. **Commit**: `[autoresearch] cycle 068: FFN Jacobian effective rank census — GPT-2`

# Validation
- Shape assertions: `W_in (3072, 768)`, `W_out (768, 3072)`, `J (768, 768)` — fail loudly if wrong.
- Random-matrix control: rank_99 >= 750 for random matrices (confirms SVD/energy logic is correct).
- No NaN in singular values; all 12 blocks produce 512 Jacobians per seed.
- Cross-seed consistency: std of mean_rank across 3 seeds < 20% of the mean.
- Correlation direction is exploratory — report rho and p-value without pre-specifying sign.

# Artifacts
- `experiments/ffn_jacobian_rank_census.py`
- `autoresearch/results/cycle_068_jacobian_rank_census.json`
- `autoresearch/results/cycle_068_jacobian_rank_census.png`

# Fallbacks
- If 512 tokens x 3 seeds is too slow (>30 min), reduce to 256 tokens and 2 seeds.
- If autograd GELU derivative fails inside hook, use manual: `sigma'(z) = 0.5*(1 + erf(z/sqrt(2))) + z/(sqrt(2*pi)) * exp(-z^2/2)`. Cross-check on 10 samples first.
- If ProbeCorpus/OpenWebText fails, fall back to `data/shakespeare.txt` (note corpus difference).
