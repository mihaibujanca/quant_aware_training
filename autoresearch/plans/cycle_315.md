# Cycle 315 Plan — Item 325

## Item 325 (verbatim Hypothesis)
cycle 313 showed calibrated SVD-LLM init at b1 r=2 gives 56.70 PPL (vs uncentered 57.66, delta -0.96, 12% of the retrain gap to cycle 280's 49.03). Open question: does the better init translate into a better RETRAIN ceiling, or does Adam erase the init difference because it converges to the same trained-optimum basin regardless of starting point? Three possibilities: (a) retrain from calibrated init < 49.03 → the SVD-LLM basis is genuinely closer to the optimal trained basis and the improvement survives retrain; (b) retrain from calibrated init == 49.03 (within 0.5 PPL) → Adam erases the init difference, basis is basin-independent, cycle 280's init choice was irrelevant to the final result; (c) retrain from calibrated init > 49.03 → calibrated init is in a worse basin (unlikely but possible if the Σ^(1/2) whitening distorts the gradient geometry). **Setup**: GPT-2 small block 1 c_proj (3072, 768); two init variants: (i) uncentered `A = U[:, :2] * S[:2]`, `B = Vh[:2, :]`, bias from W_proj_1.bias (cycle 280 recipe); (ii) calibrated SVD-LLM `Σ_l = X^T X / N` on 200×128 calib, `A_l = Σ_l^(1/2)` via eigh, `SVD(A_l @ W) = U_w S_w Vh_w`, `W_r = A_l_pinv @ (U_w[:, :2] @ diag(S_w[:2]) @ Vh_w[:2, :])`; decompose `W_r = A_cal @ B_cal` via SVD(W_r): `A_cal = U_r[:, :2] * S_r[:2]` frozen, `B_cal = Vh_r[:2, :]` trainable. Adam LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore, SEEDS=[42, 137, 2024]. Verdict: `delta_retrain ≤ -1 PPL` → CONFIRMED; `|delta_retrain| ≤ 0.3 PPL` → REFUTED; else INCONCLUSIVE.

## Dependencies
- `experiments/no_retrain_laser_baseline_cycle313.py` — calibrated Σ capture + eigh sqrt.
- `experiments/block1_lowrank_cycle280.py` — LowRankCProj + joint Adam loop + best-state restore.

## Approach
1. New script `experiments/calibrated_svd_retrain_cycle315.py` combining the two templates. Same wikitext-2 val slice as cycles 280/313 (200 calib + 100 eval, SEQ_LEN=128).
2. Uncentered factors (cycle 280 path): SVD(W) → `A2=U[:, :2]·S[:2]`, `B2=Vh[:2, :]`, bias from W_proj_1.bias.
3. Calibrated factors (cycle 313 path): hook b1 c_proj inputs, Σ=X^TX/N, A=Σ^(1/2) + A_pinv via eigh, W_r = A_pinv @ truncate(SVD(A@W), r=2); SVD(W_r) → `A_cal=U_r[:, :2]·S_r[:2]` frozen, `B_cal=Vh_r[:2, :]` trainable.
4. Cross-check init PPLs in place: uncentered==57.66±0.01 (cycle 280), calibrated==56.70±0.01 (cycle 313).
5. Retrain each variant × 3 seeds: fresh GPT-2, LowRankCProj at b1, freeze rest, Adam LR=1e-4 N_STEPS=4000 EVAL_EVERY=100, best-state restore. Compute `delta_retrain = mean(cal) - mean(unc)`, verdict per item 325.

## Files
`experiments/calibrated_svd_retrain_cycle315.py` (new); `autoresearch/results/cycle_315_calibrated_svd_retrain.json`, `_summary.md`, `_manifest.txt`, `_exec_result.txt`; `autoresearch/logs/cycle_315_run.log`, `autoresearch/audits/cycle_315.md`, `autoresearch/results/cycle_315_audit.txt`.

## Config
TARGET_BLOCK=1; RETRAIN_RANK=2; N_CALIB=200; N_EVAL=100; SEQ_LEN=128; DEVICE=mps; N_STEPS=4000; LR=1e-4; EVAL_EVERY=100; SEEDS=[42,137,2024]; trainable per fit = 2·768+768 = 2304 (same as cycle 280). Variants = 2 × 3 seeds = 6 fits. Wall ≈ 24 min.

## Verification
- Float PPL within 0.5 of 53.895. Per variant: trainable == 2304 asserted; |restored-best|<1e-3 asserted.
- Cross-check: uncentered init PPL within 0.01 of 57.664; calibrated init PPL within 0.01 of cycle 313 calibrated b1 r=2.
- Uncentered retrain best_mean should reproduce cycle 280's 49.03 within 0.1.

## Verdict logic (item 325 verbatim)
CONFIRMED iff `delta_retrain = mean(calibrated) - mean(uncentered) ≤ -1 PPL`. REFUTED iff `|delta_retrain| ≤ 0.3 PPL`. INCONCLUSIVE otherwise.

## Baselines to record
float 53.8949; uncentered b1 r=2 init=57.664 (cycle 280); calibrated b1 r=2 init=56.70 (cycle 313); cycle 280 b1 r=2 retrain best_mean=49.03.

# APPROVE
