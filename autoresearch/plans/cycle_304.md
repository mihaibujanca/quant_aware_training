# Cycle 304 Plan — Item 314: E_kept_top_k curve per block

## Item 314 (verbatim Hypothesis)
k₉₅ ∈ [3, 10] at all 12 blocks; concentrator blocks (1, 2, 10, 11) reach k₉₅ ≤ 5; mid-block 8 needs k₉₅ ∈ [8, 15]; block 0 (high eff_rank 71.93, cycle 278) needs k₉₅ ≥ 20. Together with cycle 301/302's rank-2 floor, this pins the true intrinsic rank of the curvature residual.

## Dependencies
- `experiments/curvature_residual_geometric.py` — `UnifiedCurvatureFramework`, `panel_A_projection(l, rank)`.
- `experiments/unified_curvature_cycle302.py` — driver pattern for all 12 blocks; calib 200 wikitext-2 val × 128 tok seed 42.
- `experiments/curvature_residual_A_proj_cycle301.py` — style reference for E_kept measurement.
- Cycle 302 panel 4 (E_kept_top2 per block): `[0.5902, 0.7463, 0.9190, 0.8649, 0.8426, 0.8109, 0.7197, 0.6622, 0.4722, 0.5356, 0.8374, 0.9803]`.

## Approach (no training)
1. New script `experiments/e_kept_curve_cycle304.py`. Reuse cycle 302 calib: 200 wikitext-2 val × 128 tok (seed 42), GPT-2 small, DEVICE=mps.
2. Instantiate `UnifiedCurvatureFramework(sigma=F.gelu(z, approximate='tanh'), sigma_lin=F.relu, blocks=range(12), calib_ids, device, A_factors=None)`, `capture_inputs()` once.
3. For each block l ∈ {0..11}: `compute_Dy(l)` → `W_proj, Dy`. SVD `W_proj = U diag(S) Vh` on CPU float32. For each k ∈ K_GRID (see below): `Dy_kept_k = Dy @ Vh[:k, :].T`; `E_kept_top_k(l) = ||Dy_kept_k||_F^2 / ||Dy||_F^2`. Find `k_95(l) = min k s.t. E_kept_top_k(l) >= 0.95`. Cross-check `k=2` matches cycle 302 value to 4 decimals. `free_block(l)`.
4. `K_GRID = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 500]` (fibonacci-like, covers 1..500 coarsely; 500 < 768). Also sample k ∈ {10, 20, 50, 100} as additional coarse checkpoints for the brief constraint.
5. Plot per-block E_kept_top_k curve (log-x) with 0.95 horizontal line; annotate k_95 per block.
6. Verdict per item 314 verbatim thresholds below.

## Verdict logic (item 314 verbatim)
- **CONFIRMED** iff `k₉₅ ≤ 10` at `≥ 8` of 12 blocks AND `k₉₅ ≤ 5` at ALL 4 concentrator blocks {1, 2, 10, 11}.
- **REFUTED** iff `k₉₅ ≥ 50` at `≥ 6` of 12 blocks.
- **INCONCLUSIVE** otherwise.

## Files
- New: `experiments/e_kept_curve_cycle304.py`, `autoresearch/results/cycle_304_curve.json`, `cycle_304_curve.png`, `cycle_304_summary.md`, `cycle_304_manifest.txt`, `cycle_304_exec_result.txt`, `cycle_304_audit.txt`, `autoresearch/audits/cycle_304.md`, `autoresearch/logs/cycle_304_run.log`.
- Updated: `autoresearch/agenda.md`, `autoresearch/logs/experiments.tsv`, `autoresearch/STATUS.md`.

## Config
N_CALIB=200, SEQ_LEN=128, N_TOKENS=25600, SEED=42, BLOCKS=range(12), DEVICE="mps" if available else "cpu", linalg on CPU float32. Trainable params = 0.

## Verification
- Float PPL calib match cycle 258/302 reference 45.0637 within 0.5 (assert).
- `W_proj.shape == (3072, 768)` per block; `Vh.shape == (768, 768)` (full right SVs).
- At k=2, `E_kept_top_k(l)` matches cycle 302 panel4 to 4 decimals per block (assert within 1e-4).
- `k_95(l)` monotone-in-k by construction (cumulative); sanity check `E_kept` non-decreasing across K_GRID.

## Baselines
- Cycle 302 E_kept_top2 per block (reference for k=2 column): `[0.5902, 0.7463, 0.9190, 0.8649, 0.8426, 0.8109, 0.7197, 0.6622, 0.4722, 0.5356, 0.8374, 0.9803]`.
- Isotropic null k/768: k=2 → 0.00260, k=10 → 0.013, k=100 → 0.130.
- Concentrator blocks (item 314 verbatim): {1, 2, 10, 11}. Mid-block: 8. High eff_rank block: 0 (eff_rank 71.93 from cycle 278).

# APPROVE
