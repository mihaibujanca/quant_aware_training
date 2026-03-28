# Task
Item 82: Spectral compression (Jacobian SVD basis) vs channel selection (topk) — GPT-2 FFN. Compare data-conditioned Jacobian SVD projection against topk+lstsq at matched budget for single-block swap PPL on blocks [1, 4, 7, 11].

# Hypothesis
Weight SVD lost to topk+lstsq (cycle 026), but ignores input distribution. The mean Jacobian J̄ = mean_x(W_out @ diag(GELU'(z(x))) @ W_in) captures the data-conditioned output subspace. If FFN output concentrates in a low-rank subspace (rank_99 ≈ 434–510, cycle 068), projecting onto top-k left SVs of J̄ should outperform axis-aligned topk. Note: item 81 invalidated mean Jacobian as a rank estimator, but item 82's SVD basis question is independently motivated — we test the basis quality empirically here.

# Implementation Notes
- **Conv1D transposition**: GPT-2 `c_fc.weight` is (768, 3072) and `c_proj.weight` is (3072, 768) — Conv1D stores weights as (in, out). Use `c_fc.weight.T` (3072×768) as W_in and `c_proj.weight.T` (768×3072) as W_out when computing J(x) = W_out @ diag(GELU'(z)) @ W_in.
- **GELU derivative**: Use `torch.autograd.grad` on `torch.nn.functional.gelu(z, approximate='tanh')` to get GELU'(z). Do NOT hand-code the derivative — GPT-2 uses tanh-approx GELU and manual formulas are error-prone.

# Steps
Script: `experiments/ffn_jacobian_svd_compression.py`

1. **Setup**: Load GPT-2. Probe corpus: Shakespeare, seq_len=128, 1024 tokens (≥2× overdetermined for k=512 lstsq). Eval corpus: 200 sequences, fixed seed=0. Seeds [42, 123, 7] for probe corpus construction.

2. **Mean Jacobian** (per block, per seed): Hook `mlp.c_fc` to capture input x and pre-activation z=x@W_in.T+b. For 512 tokens (subsampled from 1024 probe), compute J(x) = W_out @ diag(GELU'(z)) @ W_in per token. Average → J̄. SVD → U, S, Vh. Store U_k (768×k, k=512).

3. **Four methods** (per block, per seed, k=512):
   - **(a) topk+lstsq** (baseline): existing pipeline, hidden_fraction=1/6.
   - **(b) SVD-only**: y_approx = U_k @ U_k^T @ y_float. Pure projection, no lstsq.
   - **(c) SVD+lstsq**: features = U_k^T @ y (1024×k). lstsq refit B (768×k) over all 1024 probe tokens. Apply B @ U_k^T @ FFN(x) at eval.
   - **(d) Random orthogonal** Q_k (768×k via QR) + lstsq refit over 1024 tokens. Worst-case null baseline.

4. **Eval**: Single-block swap PPL on eval set. SVD/random methods use `register_forward_hook` on `mlp`. Report param counts per method.

5. **Output**: JSON `autoresearch/results/cycle_071_jacobian_svd_basis.json` with per-block, per-seed, per-method PPL + param counts. Plot: grouped bar chart.

Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/ffn_jacobian_svd_compression.py`

# Validation
- **Confirmed if**: SVD+lstsq outperforms topk+lstsq by ≥3% relative PPL, mean across 3 seeds, in ≥2/4 blocks.
- **Refuted if**: topk+lstsq wins in all 4 blocks (mean across seeds).
- **Inconclusive if**: mixed wins or all deltas <3%.
- Sanity: SVD-only ≥ random in all blocks; lstsq refit improves over SVD-only; topk+lstsq PPL matches prior cycles.

# Artifacts
- `experiments/ffn_jacobian_svd_compression.py`
- `autoresearch/results/cycle_071_jacobian_svd_basis.json`
- `autoresearch/results/cycle_071_jacobian_svd_basis.png`
