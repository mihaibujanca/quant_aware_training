# Cycle 298 Plan — Item 308: SDF rank census + LASER replication on 3D mesh fitting

## Item 308 (verbatim Hypothesis)
Hidden eff_rank converges to 3 (matching input dim). Gradient ensemble eff_rank ≈ 2 (matching the surface manifold dim). Rank-3 weight retrain matches float SDF MSE. Rank-2 fails to recover the surface fully (loses normal-direction information).

## Dependencies
- `aleph/mesh_datasets.py` (`load_mesh`, `sample_sdf_points` — verified working: torus mesh 2048v/4096f, SDF range [-0.23, 1.09]).
- `aleph/geometric_compression.py` (`jacobian_rank_profile`, line 374 — gradient SVD via `aleph.differential.gradient`).
- `aleph.models.MLP` (input=3, hidden=256, depth=4, output=1, ReLU).

## Approach
1. Build `experiments/sdf_rank_laser_cycle298.py`. Mesh = `torus` (item 308 says bunny "or similar"; torus is procedural and deterministic, no asset risk; surface is 2D manifold in 3D ambient).
2. Sample 30,000 SDF points (50% surface+noise 0.01, 50% volume). Random 80/20 train/held split via `torch.Generator(seed=42)`. Same split fixed across the entire cycle.
3. Build `MLP(3, 256, 1, depth=4, ReLU)`. Train Adam lr=1e-3, MSE loss, full-batch on TRAIN until held-out MSE plateau OR 5000 epochs (whichever first). Snapshot best held-out MSE state.
4. **Hidden eff_rank census** (cycle 295/296 formula: `exp(-Σ p_i log p_i)` over eigvalsh of Y^T Y / N): collect post-ReLU activations on TRAIN at all 4 hidden layers. Report eff_rank per layer.
5. **Gradient ensemble**: call `jacobian_rank_profile(model, X_train)` — returns `effective_rank`, `rank_90`, `rank_95`, `rank_99`, `singular_values`.
6. **Layer selection for LASER**: pick `l*` = argmin eff_rank over hidden layers 1..3 (skip layer 0 because input weight is (256, 3), not 256x256). All hidden weights for layers 1..3 are (256, 256).
7. **LASER sweep**: for r ∈ {2, 3, 5, 10}:
   a. **NO-RETRAIN**: SVD truncate `W_l*` at rank r (`W_r = U[:, :r] @ diag(S[:r]) @ Vh[:r, :]`); eval held-out SDF MSE; restore.
   b. **RETRAIN**: replace `model.layers[2*l*]` with `LowRankLinear(A=U[:, :r]*S[:r] frozen (256, r), B=Vh[:r, :] trainable (r, 256), bias trainable (256))`. Freeze all other params. Adam lr=1e-3 on (B, bias) for 500 full-batch MSE steps on TRAIN. Eval held-out every 25 steps; track best held-out MSE; restore best (B, bias). Restore the original Linear afterward and assert post-restore held MSE matches float baseline within 1e-9.
8. Verdict per item 308 (both empirical thresholds must hold for CONFIRMED).

## Files
- New: `experiments/sdf_rank_laser_cycle298.py`
- New: `autoresearch/results/cycle_298_sdf_rank_laser.json`
- New: `autoresearch/results/cycle_298_summary.md`
- New: `autoresearch/logs/cycle_298_run.log`
- New: `autoresearch/results/cycle_298_manifest.txt`, `cycle_298_exec_result.txt`, `cycle_298_audit.txt`, `autoresearch/audits/cycle_298.md`
- Updated: `autoresearch/agenda.md`, `autoresearch/logs/experiments.tsv`, `autoresearch/STATUS.md`

## Config
HIDDEN=256, DEPTH=4, INPUT_DIM=3, N_POINTS=30000 (24000 train / 6000 held), surface_ratio=0.5, surface_noise=0.01, MESH=torus, LR_TRAIN=LR_RETRAIN=1e-3, MAX_EPOCHS=5000, RETRAIN_STEPS=500, EVAL_EVERY=25, RANKS={2,3,5,10}, SEED=42 (single seed — SDF training is deterministic full-batch, no need for seed sweep within 15-min cap; cycle 295/296 used 3 seeds because the 2D testbed had high cross-seed variance).

## Verdict logic (item 308 verbatim)
- **CONFIRMED** iff (a) hidden output-covariance eff_rank ∈ [2.5, 3.5] (mean across hidden layers 1..3 post-training) AND (b) gradient ensemble eff_rank ∈ [1.8, 2.5] AND (c) rank-3 retrain held-out MSE ≤ 1.1× float MSE.
- **REFUTED** iff hidden eff_rank stays ≥ 8 OR rank-3 retrain MSE ≥ 2× float MSE.
- **INCONCLUSIVE** otherwise.

## Verification
- Trainable param count of MLP(3, 256, 1, depth=4) asserted at runtime.
- Pre/post LASER restore: sanity assert `eval_held_mse(restored_model) - float_held_mse < 1e-9`.
- LowRankLinear trainable count `r*256 + 256` asserted.
- Verify task performance (CLAUDE.md rule): require float held-out MSE < 0.01 (SDF on unit cube — this is a low bar; if not met, treat run as inconclusive due to undertraining).

## Baselines
- Cycle 280 GPT-2 block 1 rank-2 retrain (PPL 49.03 vs float 53.895).
- Cycle 296 spirals rank-2 retrain held-out 0.8207 ± 0.0303 (gap +0.63%).
- Random 2D subspace null in R^256 ≈ 2/256 ≈ 0.008.

# APPROVE
