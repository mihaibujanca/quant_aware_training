# Task
Item 89: SVD vs topk vs random basis at aggressive compression (k/n <= 0.3) — GPT-2 FFN. Test whether Jacobian SVD basis outperforms random orthogonal basis when retaining only 10-30% of output directions.

# Hypothesis
- **Confirmed if**: SVD+lstsq outperforms random+lstsq by >= 5% relative PPL at k/n <= 0.3 in >= 3/4 blocks (mean across 3 seeds).
- **Refuted if**: random+lstsq matches SVD+lstsq within 3% relative PPL at all tested k/n in all 4 blocks.
- **Inconclusive if**: SVD wins at k/n=0.1 but not 0.2-0.3, or varies across blocks (< 3 blocks).

# Pre-flight
- P1 (data): `data/shakespeare.txt` exists (probe corpus). Eval corpus from tokenizer (seed=0).
- P2 (deps): `experiments/ffn_jacobian_svd_compression.py` — cycle 071 script with mean Jacobian SVD, random QR basis, topk+lstsq, and per-block eval hooks. All 4 methods implemented and tested.
- P3 (ground truth): Cycle 071 baseline PPL ~121.5. At k/n=0.67: SVD+lstsq ~121.8, random+lstsq ~122.3, topk+lstsq ~126.2. At lower k/n, PPL will increase substantially. Expect PPL > 200 at k/n=0.1.
- P4 (compute): 3 seeds x 4 blocks x 3 k/n x 3 methods = 108 evals. Mean Jacobian computation ~30s/block/seed. Lstsq trivial. Eval ~3s each. Total ~25min on CPU.

# Implementation Notes
- **Conv1D transposition**: GPT-2 `c_fc.weight` is (768, 3072), `c_proj.weight` is (3072, 768). Use `.T` for standard (out, in) layout when computing Jacobian.
- **GELU derivative**: Use `torch.autograd.grad` on `F.gelu(z, approximate='tanh')`. Do NOT hand-code.
- **Mean Jacobian**: J_bar = mean_x(W_out @ diag(GELU'(z(x))) @ W_in). SVD of J_bar -> U_k (768 x k).
- **lstsq overdetermination**: At k=230 (worst case), p=231 (with bias). N=1024 tokens -> N/p=4.4. Safe per MP3 (need >= 2x). At k=77, N/p=13.3. Very safe.

# Steps
Script: `experiments/ffn_aggressive_basis_compression.py` (new, adapted from cycle 071 script)

1. **Setup.** Load GPT-2. Seeds [42, 123, 7]. Probe corpus: Shakespeare, seq_len=128, 8 sequences (1024 tokens). Eval corpus: 200 sequences, fixed seed=0. k_values = [230, 154, 77] (k/n = 0.3, 0.2, 0.1). Blocks [1, 4, 7, 11].

2. **Mean Jacobian SVD** (per block, per seed). Reuse cycle 071 approach: hook mlp.c_fc to capture input x and pre-activation z. Compute J(x) = W_out @ diag(GELU'(z)) @ W_in per token (512 probe tokens). Average -> J_bar. SVD -> U, S, Vh. Store U for all k values (take first k columns).

3. **Three methods** (per block, per seed, per k):
   - **(a) topk+lstsq**: Select top-k hidden channels by activation magnitude. Collect hidden[selected] and output samples. lstsq refit W_down (768 x k).
   - **(b) SVD+lstsq**: Project output onto U_k basis. Features = U_k^T @ y (1024 x k). lstsq refit B (768 x k). Compressed output = B @ U_k^T @ FFN(x).
   - **(c) random+lstsq**: Random orthogonal Q_k (768 x k via QR of randn). Same lstsq as SVD. Seed-dependent random basis (use seed+block_idx for reproducibility).

4. **Eval.** Single-block swap PPL for each (method, k, block, seed). Use register_forward_hook on mlp to intercept and replace output.

5. **Output.** JSON: `autoresearch/results/cycle_077_aggressive_basis.json`. Structure: {baseline_ppl, k_values, per_kn: {0.3: {per_block: {1: {seed: {method: ppl}}}}}}. Plot: 3-panel figure (one per k/n), grouped bars (3 methods x 4 blocks), error bars from 3 seeds. Save to `autoresearch/results/cycle_077_aggressive_basis.png`.

Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/ffn_aggressive_basis_compression.py`

# Validation
- Baseline PPL ~121.5 (must match cycle 071 within 1%).
- All compressed PPL > baseline (compression always degrades).
- PPL increases as k/n decreases (monotonic degradation).
- Cross-seed std < 20% of mean per condition (if higher, flag instability).
- At k/n=0.3, results should extrapolate from cycle 071 trend.

# Artifacts
- `experiments/ffn_aggressive_basis_compression.py`
- `autoresearch/results/cycle_077_aggressive_basis.json`
- `autoresearch/results/cycle_077_aggressive_basis.png`

# Fallbacks
- If all methods blow up at k/n=0.1 (PPL > 1000): report the k/n threshold where methods diverge. Focus analysis on k/n=0.2-0.3.
- If topk+lstsq is undefined at k/n=0.1 (only 77 hidden channels from 3072): topk selection still works, just picks 77 channels. Proceed.
- If SVD computation OOMs: reduce probe tokens from 512 to 256 for Jacobian averaging.
