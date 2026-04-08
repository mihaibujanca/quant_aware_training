# Cycle 306 Plan — Item 316: Block 8 deep dive (curvature residual)

## Item 316 verbatim Hypothesis
block 8's flat W_proj spectrum is the dominant factor — top-2/top-5 ratio is ≤0.3× the concentrator blocks' ratios. If confirmed, block 8 should NOT be added to any multi-block LASER stack (cycle 315) — its curvature residual has nowhere to hide.

## Dependencies
- `experiments/curvature_residual_geometric.py::UnifiedCurvatureFramework` (cycle 302).
- `experiments/curvature_residual_A_proj_cycle301.py` (style reference for E_kept measurement).
- Cycle 302 panel 4 E_kept_top_2: `[0.5902, 0.7463, 0.9190, 0.8649, 0.8426, 0.8109, 0.7197, 0.6622, 0.4722, 0.5356, 0.8374, 0.9803]`.
- Cycle 304 k₉₅ dense per block: `[445, 348, 63, 131, 152, 228, 325, 411, 511, 542, 397, 2]`.
- Cycle 278 eff_rank b8 = 291.41.

## Approach (no training)
1. New script `experiments/block8_deep_dive_cycle306.py`. Reuse calib slice: 200 wikitext-2 val × 128 tok (seed 42), GPT-2 small, DEVICE=mps, linalg CPU float32.
2. Instantiate `UnifiedCurvatureFramework(sigma=GELU tanh, sigma_lin=ReLU, blocks=[1,2,7,8,9,10,11], calib_ids, device)`. `capture_inputs()` once. Primary triple = {7, 8, 9} per user instruction; concentrator blocks {1, 2, 10, 11} included only because the item-316 verdict thresholds compare against the concentrator median (W_proj ratio AND δ eff_rank).
3. **For each block l ∈ {1, 2, 7, 8, 9, 10, 11}:** call `compute_Dy(l)` → `(Dy ∈ R^(N,768), W_proj ∈ R^(3072,768))`.
   - **Δy spectrum:** SVD `Dy = U_y diag(S_y) Vh_y` (N, 768). Record full 768-vector `S_y`. Top-1 energy fraction = `S_y[0]^2 / sum(S_y^2)`. k₉₅(Δy) and k₉₉(Δy).
   - **Per-token norms:** `||Δy_t||_2` for all N tokens; report mean, std, q05/q25/median/q75/q95, max.
   - **W_proj spectrum:** SVD `W_proj = U_w diag(S_w) Vh_w`, full 768 SVs. top1/top2/top5/top10/top20.
   - **Top-2/top-5 right SV ratio of W_proj:** `s_w[1]/s_w[4]` per item-316 verbatim "top-2/top-5 ratio".
   - **Cosine top-5(Δy) vs top-5(W_proj):** `M = Vh_y[:5] @ Vh_w[:5].T` (5,5); record absolute entries and SVD `cos = svdvals(M)` (5,) — top-5 cos sum, top-1 cos.
   - `free_block(l)` to release CPU memory between blocks.
4. **Compare 3 blocks side by side:** print/dump tables with all metrics for blocks 7, 8, 9.
5. Verdict per item 316 verbatim thresholds.

## Files
- New: `experiments/block8_deep_dive_cycle306.py`, `autoresearch/results/cycle_306_block8_deep_dive.json`, `autoresearch/results/cycle_306_summary.md`, `autoresearch/results/cycle_306_panels.png`, `autoresearch/results/cycle_306_manifest.txt`, `autoresearch/results/cycle_306_exec_result.txt`, `autoresearch/audits/cycle_306.md`, `autoresearch/results/cycle_306_audit.txt`, `autoresearch/logs/cycle_306_run.log`.
- Updated: `autoresearch/agenda.md`, `autoresearch/logs/experiments.tsv`, `autoresearch/STATUS.md`.

## Config
N_CALIB=200, SEQ_LEN=128, N_TOKENS=25600, SEED=42, BLOCKS=[7,8,9], DEVICE="mps" if available else "cpu", linalg CPU float32, **trainable_params=0**.

## Verification
- Float PPL on calib slice match cycle 302/304 reference 45.0637 within 0.5 (assert).
- For each block: `W_proj.shape == (3072, 768)`; `Vh_w.shape == (768, 768)` (full); `Vh_y.shape == (768, 768)` after SVD of `Dy ∈ R^(N, 768)` with N=25600.
- Sanity: `E_kept_top_2(b8)` derived from `Dy @ Vh_w[:2].T` matches cycle 302 ref 0.4722 within 1e-4 (assert); same check for blocks 7 (0.6622) and 9 (0.5356).
- Δy SVD eff_rank computed as `(sum S_y^2)^2 / sum(S_y^4)`; cross-check k₉₅(Δy) with cycle 304 ref `[b7: 411, b8: 511, b9: 542]` within 5%.

## Verdict logic (item 316 verbatim)
- **CONFIRMED** iff (a) block 8 W_proj top-2/top-5 SV ratio ≤ 0.5× the median concentrator block's ratio AND (b) block 8 Δy eff_rank ≥ 1.5× the median concentrator block's δ eff_rank — multiplicatively explains E_kept_top_2(b8)=0.47.
- **REFUTED** iff neither (a) nor (b) holds.
- **INCONCLUSIVE** otherwise.
- Concentrator reference: median over {1, 2, 10, 11} (per item 314/304/316 family).

# APPROVE
