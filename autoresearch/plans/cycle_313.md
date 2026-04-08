# Cycle 313 Plan — Item 323

## Item 323 (verbatim Hypothesis)
the LASER paper (Sharma et al., 2024) and SVD-LLM (Yuan et al., 2024) claim **no-training** rank-r SVD truncation of c_proj works at small accuracy cost. Cycle 280's no-retrain init was 57.66 PPL at b1 r=2; trained Adam closed it to 49.03 (+8.63 retrain delta). Question: is the 8 PPL gap from Adam (a) a feature — the SVD-truncation init is genuinely 8 PPL above the trainable rank-2 floor, or (b) a bug — our SVD init is suboptimal (e.g. wrong calibration data, wrong centering, wrong row vs column truncation, no per-token weighting). **Setup**: For each block in {b0, b1, ..., b11} and rank r ∈ {1, 2, 4, 8, 16, 32, 64}: compute c_proj_l SVD on CPU float32, truncate to rank r (`W_r = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]`), replace `c_proj_l.weight ← W_r` in-place, eval PPL on the cycle 280/305 100×128 wikitext-2 val slice. Report `no_retrain_ppl[12, 7]` matrix (84 cells). Cross-check single-block cells against published cycle 280/285/288/289/292 init PPLs (must match within 0.1). Run two parallel variants: (i) **uncentered** (cycle 280 default — straight SVD of W); (ii) **calibrated** (centre activations: SVD of `W @ E[x x^T]^{1/2}` where E[x x^T] is computed on 200 calib seqs — the SVD-LLM "weighted SVD" formulation). **Verbatim conditions**: ≥3 blocks where (calibrated, r=2) no-retrain PPL ≤ trained Adam best by ≥2 PPL → CONFIRMED; calibrated and uncentered no-retrain PPLs match within 0.5 at all blocks → REFUTED; else INCONCLUSIVE.

## Dependencies
- `experiments/block1_lowrank_cycle280.py` — template for SVD truncation, eval slice, LowRankCProj. Cycle 280 b1 r=2 noretrain=57.664, retrain=49.03 (+8.63 gap). Cycles 280/285/288/289/292 published init PPLs to cross-check.
- `experiments/block11_lowrank_cycle270.py` — template for block 11.

## Approach
1. New script `experiments/no_retrain_laser_baseline_cycle313.py`. Load GPT-2 small + wikitext-2 val (200 calib + 100 eval × 128 tok = same slice as cycles 280/305).
2. **Uncentered variant**: For each l ∈ {0..11}, compute SVD(c_proj_l.weight) on CPU float32 (HF Conv1D layout, weight shape (3072, 768)). For each r ∈ {1, 2, 4, 8, 16, 32, 64}: W_r = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]; replace c_proj_l.weight in-place; eval PPL on 100×128 slice; restore original weight; record. 12×7 = 84 cells.
3. **Calibrated variant** (SVD-LLM "weighted SVD"): for each l, capture inputs to c_proj_l via forward pre-hook on the c_proj_l module across the 200 calib seqs (200×128=25600 tokens, each row in R^3072). Compute Σ_l = E[x x^T] = (X^T X) / N ∈ R^{3072×3072}. Symmetric eigendecomposition Σ_l = V D V^T → A_l = V diag(sqrt(max(D, 0))) V^T (matrix square root, symmetric). SVD of (A_l @ W) gives U_w, S_w, Vh_w; rank-r weighted truncation: W_r = pinv(A_l) @ (U_w[:, :r] @ diag(S_w[:r]) @ Vh_w[:r, :]). Replace, eval, restore. 84 cells.
4. Save 84 cells × 2 variants in JSON. Write summary tables.
5. Verdict per item 323 verbatim. Use cycle 280/284/285/288/289/291/292 retrain anchors as the "trained Adam best" reference; compute (calibrated_r2_ppl - retrain_anchor) per block and count blocks where calibrated r=2 ≤ retrain_anchor − 2.

## Files
- `experiments/no_retrain_laser_baseline_cycle313.py` (new)
- `autoresearch/results/cycle_313_no_retrain_laser.json`, `_summary.md`, `_manifest.txt`, `_exec_result.txt`
- `autoresearch/logs/cycle_313_run.log`, `autoresearch/audits/cycle_313.md`, `autoresearch/results/cycle_313_audit.txt`

## Config
N_CALIB=200, N_EVAL=100, SEQ_LEN=128, DEVICE=mps (CPU for SVD/eigh), SEED=42, GPT-2 small, wikitext-2 val. RANKS=[1,2,4,8,16,32,64], BLOCKS=[0..11]. Trainable params = 0 (no retraining).

## Verification
- float PPL ≈ 53.895 (cycle 270/280/284/305/310 ref) within 0.5.
- Each restore brings PPL back to float within 1e-3.
- Per-(block) uncentered cells cross-check against cycle 280/285/288/289/292 published init PPLs within 0.1: cycle 280 b1 r=2 = 57.664; cycle 285 b2 r=2 = 72.3808; cycle 288 b3; cycle 289 b10; cycle 292 b1 r=1 = 61.66 (or use cycle 280 single-r seeds).
- Σ_l symmetric (|Σ_l - Σ_l^T| < 1e-5); A_l symmetric; A_l A_l ≈ Σ_l (Frobenius rel error < 1e-4).
- pinv(A_l) @ A_l ≈ I (rank-bounded, relative error tracked).

## Verdict logic (item 323 verbatim)
CONFIRMED iff |{l : (calibrated, r=2) PPL_l ≤ retrain_anchor_l − 2}| ≥ 3. REFUTED iff |calibrated_PPL_{l,r} − uncentered_PPL_{l,r}| < 0.5 for ALL (l, r). INCONCLUSIVE otherwise.

## Baselines to record
Cycle 280 b1 r=2 noretrain=57.664 retrain=49.03 (gap +8.63). Cycle 285 b2 r=2 noretrain=72.38 retrain=49.33. Cycle 288 b3 r=24. Cycle 289 b10 r=5. Cycle 270 b11 r=22 noretrain=59.683 retrain=54.47. Cycle 292 b1 r=1 retrain=50.08. Cycle 311 r=1 retrain anchors {b2, b3, b10, b11} = {49.75, 55.05, 52.91, 59.71}. Float PPL=53.895.

# APPROVE
