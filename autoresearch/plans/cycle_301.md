# Cycle 301 Plan — Item 311: Curvature residual decomposition vs cycle 280's rank-2 column space

## Item 311 (verbatim Hypothesis)
At block 1, E_kept ≤ 0.10 (the curvature residual lives ≥90% in the discarded 766-dim subspace), substantially below the random-projection null mean of 2/768 ≈ 0.0026 — "below random null" is a typo, the right reading is **E_kept ≪ ‖Δy‖²/‖Δy‖² but the absolute fraction in any 2D random subspace is ≈ 0.26%; if E_kept < 5% it's still "mostly in the complement" geometrically**, which is the geometrically meaningful claim. Specifically: E_kept ≤ 0.05 → most curvature is in the discarded subspace, LASER truncation denoises the curvature residual structurally. Expected at block 1; less expected at blocks 2 (eff_rank 1.05), 11 (eff_rank 22.49) where the kept subspace is bigger or the shape differs.

## Dependencies
- `experiments/block1_lowrank_cycle280.py` — defines A_r=(U[:, :r]·Σ[:r]) and B_r=Vh[:r, :] for rank-r SVD of `c_proj.weight (3072, 768)`. The "kept 2D output subspace in R^768" is span(Vh[:2, :]).
- `experiments/pullback_geom_variants_cycle258.py` — c_fc-input forward hook on `model.transformer.h[l].mlp.c_fc` capturing `inputs[0]` (LN2 out, 768-dim); pre-activation `preact = x_l @ c_fc.weight + c_fc.bias` (3072); GELU and ReLU per-channel.
- `experiments/cycle280_A_semantic_cycle300.py` — calib slice convention (200 wikitext-2 val × 128 tokens, seed 42, 25,600 tokens) and float-baseline assert `abs(float_ppl - 53.895) < 0.5`.

## Approach (no training)
1. New script `experiments/curvature_residual_A_proj_cycle301.py`. Load `gpt2`. Build calib: wikitext-2-raw-v1 val, first 200 sequences with len ≥ 128 (filter `len(text.strip()) > 50`), truncate to SEQ_LEN=128, identical to cycle 300. Assert `float_ppl ≈ 53.895` (use a 100-seq eval slice for the assert only).
2. For each block l ∈ {0, 1, 2, 11}:
   - Hook `model.transformer.h[l].mlp.c_fc` (forward pre-hook) capturing `inputs[0]` per calib seq → concat to `x_l ∈ R^(N, 768)`, N=25,600. Then on CPU float32: `preact = x_l @ W_fc + b_fc` (N, 3072), `delta = F.gelu(preact, approximate='tanh') - F.relu(preact)` (N, 3072), `Dy = delta @ W_proj` (N, 768) where `W_proj = c_proj.weight (3072, 768)`. No bias subtraction needed: it's a difference of two activations through the same linear map.
   - SVD `W_proj`: `U, S, Vh = torch.linalg.svd(W_proj, full_matrices=False)`. The rank-2 kept output subspace is `Vh[:2, :] ∈ R^(2, 768)` (orthonormal rows). Project: `Dy_kept = Dy @ Vh[:2, :].T @ Vh[:2, :]`. Since rows are orthonormal, `||Dy_kept||_F^2 = ||Dy @ Vh[:2, :].T||_F^2`. Compute `E_kept = ||Dy @ Vh[:2, :].T||_F^2 / ||Dy||_F^2`. Also compute `E_kept_top1 = ||Dy @ Vh[0:1, :].T||_F^2 / ||Dy||_F^2`.
   - Per-token absolute cosine of `Dy[t]` with `col(A)` (the 2-plane span(Vh[:2, :])): `c_t = ||Dy[t] @ Vh[:2, :].T|| / (||Dy[t]|| + 1e-12)`. Save the (N,) array — store mean / median / quartiles plus a histogram (50 bins) in JSON.
   - Random null: 100 random 2D orthonormal subspaces in R^768. Generator seed 20260406 (matches cycle 300 null). For each, draw `R = randn(768, 2)`, QR-orthonormalise to get Q (768, 2), compute `E_rand = ||Dy @ Q||_F^2 / ||Dy||_F^2`. Report mean / std / 95% CI (mean ± 2σ). Sanity: expected ≈ 2/768 = 0.00260 if Dy were isotropic; deviation measures Dy's anisotropy.
3. Verdict per item 311 verbatim, **block 1**:
   - **CONFIRMED** iff `E_kept(block 1) ≤ 0.05`.
   - **REFUTED** iff `E_kept(block 1) ≥ 0.30`.
   - **INCONCLUSIVE** otherwise.
   Report blocks 0/2/11 as supporting context.
4. Write `cycle_301_decomp.json` with per-block `{E_kept_top2, E_kept_top1, ||Dy||_F, ||Dy_kept_top2||_F, null_mean, null_std, null_min, null_max, cos_quartiles, cos_hist}`, plus `verdict_block_1`. Plot 4-panel histogram (one per block) of per-token |cos(Dy_t, col(A_2))|, save to `cycle_301_cos_histograms.png`.

## Files
- New: `experiments/curvature_residual_A_proj_cycle301.py`
- New: `autoresearch/results/cycle_301_decomp.json`, `cycle_301_cos_histograms.png`, `cycle_301_summary.md`, `cycle_301_manifest.txt`, `cycle_301_exec_result.txt`, `cycle_301_audit.txt`, `autoresearch/audits/cycle_301.md`, `autoresearch/logs/cycle_301_run.log`
- Updated: `autoresearch/agenda.md`, `autoresearch/logs/experiments.tsv`, `autoresearch/STATUS.md`

## Config
N_CALIB=200, SEQ_LEN=128, SEED=42, BLOCKS=[0,1,2,11], RANK=2, N_RANDOM=100, NULL_SEED=20260406, DEVICE="mps" if available else "cpu" (linalg on CPU float32). N_TOKENS=25,600. Trainable params = 0.

## Verification
- `W_proj.shape == (3072, 768)` and `Vh[:2, :].shape == (2, 768)` per block.
- `Vh[:2, :] @ Vh[:2, :].T ≈ I_2` (orthonormality, max abs off-diag < 1e-5).
- `0 ≤ E_kept ≤ 1` per block.
- `||Dy||_F > 0` (curvature is non-trivial — sanity).
- Float PPL within 0.5 of 53.895 (calib slice unchanged from cycle 300).
- Random null mean for an isotropic Dy ≈ 2/768 = 0.00260; deviation measures anisotropy and is reported, not asserted.

## Baselines
- Random 2D subspace null mean ≈ 2/768 = 0.0026 (isotropic Dy reference).
- Cycle 217: Δ̃ rank_95 ≤ 64 in 9/12 blocks (Dy rank is concentrated).
- Cycle 258: κ shift +47.7% at layer 4 under ReLU swap (Dy is operationally non-trivial).
- Cycle 280: A factor at block 1 = `(U[:, :2] * S[:2])` of c_proj SVD; eff_rank b1=1.94 (cycle 278), b2=1.05, b11=22.49.

# APPROVE
