# Task
Jacobian variance atlas for GPT-2 FFN blocks on proxy gpt2+Shakespeare.

# Hypothesis
From agenda item 19:
- **Confirmed if**: at least 4 of 24 GPT-2 blocks have L ≥ 0.85 AND Spearman correlation between L and (1 − relative perplexity increase from cycle 005) is ≥ 0.4.
- **Refuted if**: all blocks have L < 0.6, indicating that every block exhibits substantial Jacobian variation and single-affine collapse is not viable for any block.
- **Inconclusive if**: linearity index correlates with depth but not with compression damage, suggesting L measures something other than compressibility.

NOTE: GPT-2 has 12 FFN blocks, not 24. The "4 of 24" criterion scales to "2 of 12" for L ≥ 0.85 (same 1/6 fraction). The Spearman threshold remains ≥ 0.4.

# Pre-flight Results
- P1: Not a duplicate.
- P2: Depends on item 9 (done). Met.
- P3: Concrete and testable.
- P4: No prompt/program modifications.

# Steps

1. **Write** `experiments/jacobian_variance_atlas.py`:
   - Load gpt2 teacher + Shakespeare corpus
   - For each of 12 FFN blocks:
     - Get preactivation samples z from probe_ffn_geometry
     - Compute activation derivative d(x) = σ'(z(x)) for each token
     - Compute d̄ = mean(d(x))
     - Compute ||J̄||_F where J̄ = (W_out * d̄) @ W_in
     - For each token: δ = d(x) - d̄, J_diff = (W_out * δ) @ W_in, ratio = ||J_diff||_F / ||J̄||_F
     - L = 1 − mean(ratio)
   - Correlate L with (1 − relative_ppl_increase) from cycle 005
   - Save results, plot

2. **Run**: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/jacobian_variance_atlas.py`

3. **Evaluate** against criteria.

4. **Move item 19** to PAST, commit, manifest.

# Validation
- L should be between 0 and 1 for all blocks
- Layer 0 should have different L than middle layers (known hard case)
- Spearman correlation should be computed on 12 data points

# Fallbacks
- If 2048 tokens is too slow, use 256 tokens per block
- If memory issues, process one block at a time (already the default)

# Artifacts
- `experiments/jacobian_variance_atlas.py`
- `autoresearch/results/cycle_009_jacobian_atlas.json`
- `autoresearch/results/cycle_009_summary.md`
