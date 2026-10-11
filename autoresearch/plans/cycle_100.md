# Task
Low-rank Jacobian factorization for FFN compression — GPT-2. Test whether a global U @ diag(S(x)) @ V^T factorization (shared U,V, token-varying S) can compress FFN blocks at rank 64.

# Hypothesis
- **Confirmed if**: rank-64 factorization achieves <= +15% single-block PPL for >= 2/4 blocks.
- **Refuted if**: (a) median relative reconstruction error > 0.5 at r=128 (global factorization itself fails), OR (b) rank-64 > +30% PPL everywhere.
- **Inconclusive if**: rank varies wildly across blocks or tokens.

# Steps
1. Create `experiments/jacobian_lowrank_factorization_gpt2.py`. Reuse probe corpus and teacher loading from `aleph/lm_geometric_compression.py` and Jacobian computation from `experiments/ffn_jacobian_rank_census.py`.
2. **Phase 1 — Per-token rank audit**: Compute J(x) for 2048 tokens × blocks [1,4,7,11]. Report rank_95 per token. Expected >200 (cycle 068). Note but proceed to global factorization.
3. **Phase 2 — Global factorization via mean-Jacobian SVD**: Compute J_bar = mean(J(x_i), i=1..2048). SVD of J_bar (768×768) → U_full (768×768), Sigma, V_full (768×768). Truncate to rank r: U = U_full[:, :r] (768×r), V = V_full[:, :r] (768×r). For each token, compute oracle diagonal S(x) = diag(U^T @ J(x) @ V) (length-r vector). Measure reconstruction error: ||J(x) - U @ diag(S(x)) @ V^T||_F / ||J(x)||_F. Report median over tokens. Test r ∈ {32, 64, 128, 256}. **Refutation gate**: median relative error > 0.5 at r=128 → REFUTED.
4. **Phase 3 — S(x) predictor via ridge regression**: Fit S(x) = A @ x + b (A: r×768, b: r) using ridge regression (sklearn Ridge, CV over alpha). Train on 1536 tokens, eval on 512. Report MSE on eval set. Total predictor params: r×768 + r (e.g., 49,216 at r=64).
5. **Phase 4 — Single-block swap PPL**: For each block in [1,4,7,11], replace FFN with f_approx(x) = U @ diag(S_pred(x)) @ V^T @ x + b_ffn. Measure PPL on WikiText-2 val with 3 seeds (different 512-token eval slices). Compare vs: (a) baseline PPL (~51.66), (b) oracle S(x) PPL, (c) random U,V (orthogonal) + oracle S as null baseline, (d) topk+lstsq at equivalent compression. Report total params: 2×768×r (U,V) + r×768 + r (predictor) + r (bias absorbed). E.g., r=64: ~148K vs 4.7M original.
6. Save results JSON to `autoresearch/results/cycle_100_jacobian_factorization.json` and plot to `autoresearch/results/cycle_100_jacobian_factorization.png`.
7. Update `autoresearch/agenda.md`: move item 103 from NEXT to PAST EXPERIMENTS with outcome, cycle number, and artifact paths.
8. Commit: `[autoresearch] cycle 100: low-rank Jacobian factorization — GPT-2 FFN`

# Validation
- Phase 2 at r=768 (full rank): reconstruction error must be ~0 (sanity check).
- Oracle S(x) PPL <= predicted S(x) PPL (oracle is upper bound on factorization quality).
- Random U,V PPL should be worse than SVD U,V. If not, shared structure hypothesis is wrong.
- Baseline PPL ≈ 51.66 ± 5 (consistent with prior cycles).
- Per-token rank_95 consistent with cycle 068 (268–314 for blocks 3–11).

# Artifacts
- `experiments/jacobian_lowrank_factorization_gpt2.py`
- `autoresearch/results/cycle_100_jacobian_factorization.json`
- `autoresearch/results/cycle_100_jacobian_factorization.png`

# Fallbacks
- If mean-Jacobian SVD reconstruction error is high: try column-stacked SVD (stack J(x_i) vertically as (2048×768, 768), SVD → V from right singular vectors, U from projection).
- If S(x) predictor fails: report oracle-only results — if oracle PPL is bad, factorization itself is flawed.
- If PPL eval OOM: reduce to 1024 tokens for fitting.
