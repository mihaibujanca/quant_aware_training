# Task
MI bottleneck channel selection vs topk — GPT-2 FFN. For blocks [1, 4, 7, 11] at 6× (k=512 of 3072), compare MI-greedy forward selection + lstsq vs topk + lstsq on single-block swap PPL.

# Hypothesis
- **Confirmed if:** `MI-greedy + lstsq outperforms topk + lstsq by ≥ 5% relative PPL in ≥ 2/4 blocks.`
- **Refuted if:** `MI-greedy selects ≥ 90% the same channels as topk (the heuristic is already near-optimal).`
- **Inconclusive if:** `MI-greedy selects different channels but PPL improvement is < 3%.`

# Pre-flight Results
- **P1 (duplicate):** PASS. No prior cycle tested MI-based channel selection. Cycles 071/077 tested basis (projection matrix) — orthogonal. Cycle 053 studied region-aware merging — different mechanism. This tests the selection heuristic itself.
- **P2 (dependency):** PASS. Item 86 lists "Depends on: none."
- **P3 (concrete):** PASS. Two channel selection methods, four blocks, one metric (PPL), one cycle. Channel overlap is a counting operation.
- **P4 (program modification):** PASS. Diagnostic comparison of selection methods. Does not modify program.md or compression pipeline.
- **P5 (null baseline):** Random channel selection at 6× should yield significantly worse PPL than either MI-greedy or topk — confirms both methods capture real signal. Baseline GPT-2 PPL ≈ 57.65.
- **P6 (architecture fit):** Gaussian MI approximation is exact when activations are jointly Gaussian. GPT-2 FFN hidden activations (post-GELU) are not Gaussian, but the Gaussian proxy captures second-order structure (covariance). This is the standard approach for tractable high-dimensional MI; MINE is overkill for a selection comparison. Method fits the data shape (3072-dim hidden, 768-dim output). **Note:** Under the Gaussian model, greedy MI-maximizing channel selection is mathematically equivalent to greedy column subset selection minimizing det(Σ_{y|S}) — i.e., maximizing explained variance of y given the selected columns. This is a classical column subset selection criterion for linear regression, not a general MI estimator. The experiment tests "near-optimal column subset selection vs topk heuristic for lstsq."
- **P7 (statistical power):** 3 seeds for PPL (probe sampling). 4 blocks × 2 methods × 3 seeds = 24 PPL measurements. Channel overlap is deterministic given the covariance matrix. 8192 probe tokens for covariance estimation (N/D ≈ 2.67 for 3072-dim hidden); 4096 tokens for lstsq fitting (N/p = 8, safe).

# Steps
1. Create `experiments/mi_channel_selection.py`. Load GPT-2, hook FFN blocks [1, 4, 7, 11].
2. **Probe forward pass** (8192 tokens for covariance, 4096 tokens for lstsq): For each block, capture hidden activations h (B×T, 3072) and block output y (B×T, 768). The covariance pass needs N/D ≥ 2.67 for reliable 3072-dim estimation (8192/3072 ≈ 2.67); 4096 tokens (N/D = 1.33) is in the Marchenko-Pastur noise regime where small eigenvalues are pure noise and Schur complement rankings become unreliable.
3. **Topk selection**: Use existing `channel_scores = std_h * output_norms * (mean_h.abs() + mean_deriv.abs())` from `aleph/lm_geometric_compression.py` line 695. Select top-512 channels.
4. **MI-greedy forward selection**: Compute Σ_yy = cov(y), Σ_hy = cov(h, y) from the 8192-token covariance pass. Greedy loop: at each step i=1..512, for each candidate channel c not yet selected, compute det(Σ_residual) after adding c. Select the c that minimizes det(Σ_residual). Use Cholesky rank-1 updates for O(k²d) total instead of naive O(k³d). Specifically: Σ_residual = Σ_yy - Σ_y,sel @ Σ_sel,sel⁻¹ @ Σ_sel,y. Each new channel = rank-1 update to Schur complement. **Recompute the full Schur complement from scratch every 50 channels** to prevent accumulated numerical drift in the rank-1 updates (negligible cost: 512/50 = ~10 full recomputations).
5. **Lstsq reconstruction** (4096 tokens, separate pass): For both channel sets, compute W_approx = lstsq(h[:, selected], y). Replace FFN output weights.
6. **Single-block swap PPL** (3 seeds): For each block × method, swap the compressed FFN weights in, measure PPL on OWT eval. Also run random-512 selection as null baseline.
7. **Channel overlap**: Compute |MI_set ∩ topk_set| / 512 for each block.
8. Collect results into `autoresearch/results/cycle_091_mi_channel.json`. Summary table: block, method, PPL, ΔPPL%, overlap%.
9. Evaluate against criteria. Update `autoresearch/agenda.md`.
10. Commit: `[autoresearch] cycle 091: MI bottleneck channel selection — GPT-2`

# Validation
- Baseline PPL (no modification) ≈ 57.65 ± 1.0.
- Random-512 selection must be significantly worse than both MI-greedy and topk (confirms both heuristics beat random).
- MI-greedy covariance matrix must be PD (check condition number; if >1e10, add regularization 1e-6·I).
- Greedy selection must be monotonic: each added channel should reduce or maintain det(Σ_residual).

# Artifacts
- `experiments/mi_channel_selection.py`
- `autoresearch/results/cycle_091_mi_channel.json`

# Fallbacks
- If Cholesky rank-1 update is numerically unstable even with periodic recomputation every 50 steps, fall back to full recomputation every step (slower but guaranteed stable).
- If MI-greedy is too slow (>10 min per block), subsample to 4096 tokens for covariance estimation (N/D = 1.33; add regularization 1e-5·I to compensate).
- If all 4 blocks show <3% difference and >90% overlap, report REFUTED early (topk is near-optimal).
