# Cycle 309 Plan — Item 319: Δy-aligned rank-r LASER at block 8

## Item 319 verbatim Hypothesis
cycle 306 showed Δy at b8 is rank-1 dominated on its own SVD (top1_E ≈ 0.91, stable rank 1.20), but the standard cycle 280/284/305 LASER builds A from W_proj's left singular vectors and B from Vh_w[:r, :], leaving 53% of Δy energy unprojected at r=2. Test whether a Δy-basis low-rank replacement at r=1 and r=2 closes the b8 PPL gap that the W_proj-basis variant cannot.

## Geometric construction
At block 8, c_proj.weight ∈ R^(3072,768) is applied as `y = GELU(z) @ W_proj`. Cycle 306 reports `Δy = (GELU(z) - ReLU(z)) @ W_proj` is rank-1 dominated (top1_E=0.914) with right-singular basis Vh_y ∈ R^(768,768). Vh_y lives in the **output (residual)** space; constraining the new c_proj's row space to span(Vh_y[:r]) preserves Δy's top-r directions. Set `W_new = A_y @ B`, where `A_y = W_proj @ Vh_y[:r].T` ∈ R^(3072,r) is **frozen**, and `B` ∈ R^(r,768) is **trainable** (init = Vh_y[:r]). At init: `A_y @ B = W_proj @ (Vh_y[:r].T @ Vh_y[:r])` = orthogonal projection of W_proj's columns onto span(Vh_y[:r]). Trainable bias = c_proj.bias clone.

## Paired baseline (W_proj basis)
Cycle 280-style LASER at b8: `A_w = U_w[:, :r] * S_w[:r]` frozen, `B_w = Vh_w[:r]` trainable, same bias treatment. Run as paired baseline at r=1 and r=2 in the SAME cycle to isolate the basis change.

## Files
- New: `experiments/block8_dy_aligned_laser_cycle309.py`
- Reuses: `experiments.curvature_residual_geometric.UnifiedCurvatureFramework` (for Δy@b8), and the `LowRankCProj` template from `block11_lowrank_cycle270.py`.

## Config
GPT-2 small, wikitext-2 val, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, DEVICE=mps, FLOAT_REF=53.895 (eval slice). For Δy: 200 calib seqs × 128 = 25600 tokens at b8, single forward pass. Ranks {1, 2}. Two bases: `dy` (Vh_y) and `wproj` (Vh_w). No-retrain PPL for all 4 cells. Adam LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS={42, 137, 2024}, best-state restore. Wall budget: 4 no-retrain (+ 4 retrain × 3 seeds × ~3 min) ≈ 40 min — under 60-min cap.

## Verification
- `Vh_y` shape (768, 768); orthonormal: `Vh_y[:r] @ Vh_y[:r].T ≈ I_r` (asserted < 1e-4).
- Float PPL match cycle 270/280/305 ref 53.895 within 0.5 (asserted).
- At init, `dy` variant init PPL == `dy` no-retrain PPL (asserted < 1e-2 per seed).
- Trainable params per fit = r*768 + 768 (asserted).
- Restore best state: `restored_ppl == best_ppl` to 1e-3 (asserted per seed).

## Verdict logic (item 319 verbatim)
- **CONFIRMED** iff r=1 Δy-aligned retrain best_mean ≤ 60.0 — "Δy basis is the right basis at b8, W_proj basis is the wrong one".
- **REFUTED** iff r=1 Δy-aligned retrain best_mean ≥ 110.0 — "the b8 53% unprojected energy is not recoverable from any rank-1 cytoplasm; b8 needs higher rank regardless of basis".
- Otherwise INCONCLUSIVE.

## Outputs
- `autoresearch/results/cycle_309_block8_dy_aligned_laser.json` — full per-cell results.
- `autoresearch/results/cycle_309_summary.md` — tables (no-retrain 2×2; retrain 2×2 ± std).
- `autoresearch/results/cycle_309_manifest.txt`, `cycle_309_exec_result.txt`, `cycle_309_run.log`.

# APPROVE
