# Cycle 309 Design Review — Item 319

**Status: APPROVE**

## Plan summary
Δy-aligned rank-r LASER at GPT-2 b8. For r ∈ {1, 2}: build `W_new = A_y @ B`, where `A_y = W_proj @ Vh_y[:r].T` is frozen (3072, r), `B` trainable (r, 768) init = Vh_y[:r], bias trainable. Paired W_proj-basis baseline at the same r. Eval no-retrain PPL + 3-seed Adam retrain. Verdict per item 319 verbatim thresholds on r=1 Δy-aligned retrain best_mean.

## Design correctness
- **Geometric basis interpretation correct.** Vh_y ∈ R^(768,768) is the orthonormal basis of the Δy output space. Constraining each row of W_new to live in span(Vh_y[:r]) preserves Δy's top-r directions in the residual stream — the right reading of the verbatim phrase "B from Vh_y[:r, :]". The verbatim "A from U_y" is shape-incompatible (U_y is (N_tokens, 768)) and the brief explicitly redirects to the projection formulation `W_proj @ (Vh_y[:r].T @ Vh_y[:r])` which the plan implements via `A_y = W_proj @ Vh_y[:r].T`, `B_init = Vh_y[:r]`. ✓
- **Frozen factor choice is the principled one.** A_y captures the input-side mixing W_proj has already learned and freezes it; B is the output-space rank-r basis that Adam can rotate. Mirrors cycle 280's contract (frozen A = `U_w * S_w`, trainable B = `Vh_w`).
- **Paired baseline isolates the basis change.** Same template `LowRankCProj`, same training loop, same N_STEPS, same seeds — the only delta is the choice of A and B_init, which is exactly what item 319 asks to test ("compare against cycle 280-style W_proj-basis r=1 b8 LASER must be run as a paired baseline in the same cycle to isolate the basis change").

## Risks checked
- **Δy SVD on 25600×768 → CPU float32.** The 306 framework already chunks; reuse `compute_Dy(8)`. Memory ~150 MB, fine.
- **Vh_y orthonormality.** torch.linalg.svd returns orthonormal Vh; assert `Vh_y[:r] @ Vh_y[:r].T ≈ I_r` to 1e-4. ✓
- **MPS float32 SVD numerical drift.** SVD done on CPU per cycle 270 convention. ✓
- **Init PPL sanity.** dy-variant init PPL must equal dy no-retrain PPL (within 1e-2). Asserted per seed.
- **Best-state restore.** `restored_ppl == best_ppl` (1e-3). Asserted.
- **Wall budget.** 4 no-retrain (~40 s) + 4 retrain configs × 3 seeds × ~3 min = ~36 min training + ~5 min overhead ≈ 41 min. Under 60 min cap.

## Verdict thresholds
Item 319 verbatim: CONFIRMED ≤ 60 PPL; REFUTED ≥ 110 PPL; else INCONCLUSIVE. Both thresholds applied to r=1 Δy-aligned retrain best_mean. The W_proj r=1 baseline does not gate the verdict but informs the substantive reading.

## Plan adherence checks
- 8 H2 headings (Item 319 verbatim Hypothesis, Geometric construction, Paired baseline, Files, Config, Verification, Verdict logic, Outputs). ✓
- ≤ 45 lines (36 lines). ✓
- Item 319 verbatim Hypothesis present and unedited. ✓
- New script path: `experiments/block8_dy_aligned_laser_cycle309.py`. ✓
- Reuses `UnifiedCurvatureFramework` from cycle 302/306. ✓

## Decision
APPROVE. Proceed to Execute.
