# Cycle 316 Plan — Item 326

## Item 326 (verbatim Hypothesis)
cycle 310 found (b10, b11) is the unique destructive pair at +6.25 PPL; all 9 other pairs non-destructive. The two blocks are the LATEST cycle-282-family concentrators in the residual stream. Three candidate mechanisms to discriminate: (M1) **residual stream redundancy** — b10 and b11 write similar directions into the residual stream, and reducing both to rank-r forces them to share a limited subspace leading to cancellation/interference; (M2) **coupling via LayerNorm** — b11's LN input depends on b10's output (since b10 is the previous block's c_proj), low-rank at b10 distorts b11's LN-normalised input and breaks b11's subsequent curvature fit; (M3) **sequential Adam optimisation pathology** — joint Adam over two tightly-coupled blocks cannot escape a saddle the sequential single-block fits would avoid.

## Dependencies
- `experiments/pairwise_laser_cycle310.py` — joint pair LowRankCProj template + cycle 310 (b10,b11) joint best_mean = 60.7161 (1 seed, N_STEPS=2000, LR=1e-4, init=76.6535).
- `experiments/block10_lowrank_cycle282.py` / `experiments/block11_lowrank_cycle270.py` — single-block LowRankCProj reference (rank anchors b10 r=5 → 51.4659, b11 r=22 → 54.4687).
- `experiments/curvature_residual_geometric.py` — `compute_Dy(l)` returns per-token Δy = (σ(z) − σ_lin(z)) @ W_proj for cycle-302 framework (used for M1 angle).

## Approach
1. New script `experiments/b10_b11_deep_dive_cycle316.py`. Reuse `LowRankCProj` template (3072×r frozen A, r×768 trainable B, 768 trainable bias). Float PPL ref 53.8949; assert within 0.5.
2. **M1 (residual stream redundancy)**: forward calib (200 seqs × 128 tok) once, capture `mlp_inputs[10]`, `mlp_inputs[11]` via cycle 302 `capture_cfc_inputs`. For l ∈ {10, 11} compute Δy_l = GELU(z_l)·W_proj_l from float weights (so Δy_l matches the residual contribution of the FFN nonlinearity above linear), where z_l = ln_2(x_l) @ W_fc_l + b_fc_l, and Δy_l = GELU(z_l) @ W_proj_l. SVD each: take top-5 right singular vectors (Vh_l[:5, :768]) for b10 and top-22 for b11 (matched to cycle-282 ranks r_10=5, r_11=22). Compute (a) `principal_angle_5_22 = svdvals(Vh_10[:5] @ Vh_11[:22].T)` (5 cosines), (b) `principal_angle_5_5 = svdvals(Vh_10[:5] @ Vh_11[:5].T)` (5 cosines), (c) report mean angle in degrees from `arccos(cos)`. Threshold per item 326: if mean(principal_angle) < 30° → M1 supported.
3. **M2 (sequential Adam)**: load fresh GPT-2. Install both LowRankCProj (b10 r=5, b11 r=22). Phase A: freeze everything except {B_10, bias_10}; Adam LR=1e-4, N_STEPS=2000, EVAL_EVERY=100, best-state restore. Phase B: copy phase-A best back, freeze {B_10, bias_10}, unfreeze {B_11, bias_11}; same Adam from scratch (fresh optimizer state), N_STEPS=2000, best-state restore. Report `sequential_best_ppl` = min eval PPL across both phases (with restored-state assertion |restored - best| < 1e-3 per phase). If `sequential_best_ppl ≤ 55` → M3 confirmed (joint-Adam saddle); compare to cycle 310 joint = 60.7161.
4. **M3 (one-sided restore)**: two ablations.
   - **Ablation A** (b10 float + b11 r=22): leave block 10 c_proj at float weights, install LowRankCProj at b11 only at r=22; fresh Adam over {B_11, bias_11} for N_STEPS=2000 (matches cycle 270 setup but on 100×128 slice / SEED=42 / N_STEPS=2000 for budget parity). Report `b10float_b11r22_best_ppl`.
   - **Ablation B** (b10 r=5 + b11 float): mirror; install LowRankCProj at b10 only, leave b11 at float; same Adam budget. Report `b10r5_b11float_best_ppl`.
   - Compare both to cycle 310 joint (60.7161). If both ablations land near their cycle 310 single-block diagonals (b10≈51.47, b11≈54.47 within +1 PPL) but joint sits at 60.72, the destruction is *interactive*, not single-block; if either ablation already sits at 60+ PPL the destruction is single-block.

## Files
- `experiments/b10_b11_deep_dive_cycle316.py` (new)
- `autoresearch/results/cycle_316_b10_b11_deep_dive.json`, `_summary.md`, `_manifest.txt`, `_exec_result.txt`
- `autoresearch/logs/cycle_316_run.log`, `autoresearch/audits/cycle_316.md`, `autoresearch/results/cycle_316_audit.txt`

## Config
BLOCKS={10:5, 11:22}; LR=1e-4; N_STEPS=2000 per phase / per ablation; EVAL_EVERY=100; SEEDS=[42] (item 326 verbatim); N_CALIB=200; N_EVAL=100; SEQ_LEN=128; DEVICE=mps; gpt2; wikitext-2 val. Wall budget: M1 ~1 min (one forward + 2 SVDs); M2 ~4 min (2× 2000-step Adam); M3 ~4 min (2× 2000-step Adam single-block). Total ~10 min ≤ item-326 ~8 min hint, with margin.

## Verification
- Float PPL within 0.5 of 53.8949.
- Per LowRankCProj install: expected trainable asserted.
- Per Adam phase: |restored - best| < 1e-3.
- M1 sanity: Vh_10 @ Vh_10.T ≈ I_5 (orthonormal); Vh_11 @ Vh_11.T ≈ I_22.
- M3 ablation A init PPL must equal cycle 270 r=22 init (59.683 ± 0.05); ablation B init PPL must equal cycle 282 r=5 init (b10 r=5 noretrain ≈ 56.5).
- Cross-check: cycle 310 (b10,b11) joint best 60.7161 reproducible if SEED=42 unchanged (sanity, recorded not asserted because budget halved from possible reruns).

## Verdict logic (item 326 verbatim)
- `sequential_ppl ≤ 55` → CONFIRMED-optimisation-pathology (M3): "cycle 310 destruction was joint Adam saddle; sequential Adam or coordinate descent fixes it".
- `sequential_ppl ≥ 60` AND `cos_angle_d10_d11 < 30°` → CONFIRMED-residual-redundancy (M1): "b10 and b11 write overlapping directions; both cannot be compressed simultaneously".
- `sequential_ppl ≥ 60` AND `cos_angle_d10_d11 > 60°` → REFUTED-mechanism-unknown: "neither candidate explains it; LN coupling (M2) or higher-order dynamics".
- else INCONCLUSIVE.

## Baselines to record
float 53.8949; cycle 270 b11 r=22 single-block 54.4687; cycle 282 b10 r=5 single-block 51.4659; cycle 310 (b10,b11) joint 60.7161 (init 76.6535, SEED=42, N_STEPS=2000).

# APPROVE
