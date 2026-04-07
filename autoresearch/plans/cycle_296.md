# Cycle 296 Plan — Item 306

## Dependencies
Item 305 trained spirals/moons setup (cycle 295) — retrain inline. Reuses `aleph.datasets.make_spirals`, `aleph.models.MLP`, `sklearn.datasets.make_moons`. SVD retrain template from `experiments/block1_lowrank_cycle280.py` (LowRankCProj pattern).

## Item 306 (verbatim Hypothesis)
rank-2 retrain matches float held-out acc within 0.5% on spirals. rank-1 fails by ≥3% (need at least 2 dimensions for the spiral). On moons, rank-1 might already match (the moons separator is ~1D after a coordinate change).

## Approach
New script `experiments/spirals_moons_laser_cycle296.py`. For each dataset ∈ {spirals, moons} × seed ∈ {42, 137, 2024}: build MLP(2,32,1,depth=12,ReLU); train (cycle 295 recipe: Adam lr=1e-3 BCEWithLogits full-batch, ≥95% train acc, max 20k epochs, spirals input ×2.0). Split N=2000 into 1000 train / 1000 held-out (random_state=seed+1). Train net on TRAIN ONLY. Then identify ℓ* = argmin_ℓ eff_rank(layer ℓ) over ℓ ∈ {0..11} on TRAIN post-ReLU activations (consistent with item 305's 2D-bottleneck finding — expect ℓ* ∈ {7..11}). Take W ∈ R^(32, 32) from `model.layers[2*ℓ*]` (Linear weight, shape (32,32) since hidden_size=32 for ℓ ≥ 1). For r ∈ {1, 2, 3, 5}: SVD W = U Σ V^T (full-matrices=False); A_r = U[:, :r]·diag(Σ[:r]) (frozen, 32×r); B_r init = V^T[:r, :] (r×32, trainable); bias_r init = original layer bias (32, trainable). (i) NO-RETRAIN: write W_r = A_r @ B_r into the layer's weight, eval held-out acc, restore W. (ii) RETRAIN: replace `model.layers[2*ℓ*]` with a `LowRankLinear` module wrapping (frozen A, trainable B, trainable bias). Freeze all other params. Adam lr=1e-3 on (B, bias), 500 full-batch BCEWithLogits steps on TRAIN, eval held-out every 25 steps, track best held-out acc. Restore best B/bias and assert. Restore original layer afterward for next r.

## Files
- `experiments/spirals_moons_laser_cycle296.py` (new)
- `autoresearch/results/cycle_296_spirals_moons_laser.json`
- `autoresearch/results/cycle_296_summary.md`
- `autoresearch/results/cycle_296_manifest.txt`
- `autoresearch/results/cycle_296_exec_result.txt`
- `autoresearch/logs/cycle_296_run.log`
- `autoresearch/audits/cycle_296.md`
- `autoresearch/results/cycle_296_audit.txt`

## Config
HIDDEN=32 DEPTH=12 N=2000 (1000 train + 1000 held-out, random split per seed) LR_TRAIN=1e-3 LR_RETRAIN=1e-3 MAX_EPOCHS=20000 TRAIN_ACC_TARGET=0.95 RETRAIN_STEPS=500 EVAL_EVERY=25 RANKS=[1,2,3,5] seeds={42,137,2024} spirals(noise=0.5,n_turns=3,input_scale=2.0) moons(noise=0.1) CPU. Trainable params per retrain fit ≈ 32(r+1).

## Verification
- Net trainable param count = 11,745 asserted.
- Net train_acc ≥ 0.95 asserted before LASER (CLAUDE.md "verify task performance before analysis"); else seed marked failed and skipped.
- For each seed: float held-out acc recorded as baseline.
- ℓ* selected per (dataset, seed) by argmin of eff_rank on TRAIN post-ReLU (2 ≤ ℓ ≤ 11; skip ℓ=0 since W is (32,2) — only 2 singular values, not the comparison of interest). Recorded.
- W shape asserted (32, 32).
- No-retrain held-out acc asserted equals eval after writing W_r into layer.
- After retrain best-state restore, eval matches best held-out acc within 1e-9.
- Original layer fully restored between r values; sanity: float held-out acc unchanged after restore.

## Verdict logic (item 306 verbatim)
- CONFIRMED iff rank-2 retrain held-out acc ≥ float held-out acc − 0.5% on spirals AND no-retrain rank-2 ≥ 80% of float held-out acc.
- REFUTED iff rank-2 retrain held-out gap > 5% from float on spirals.
- INCONCLUSIVE between.
Aggregate per dataset: mean across 3 seeds. Spirals drives the verdict; moons reported for context (rank-1 prediction).

## Baselines
Cycle 280 block 1 rank-2 retrain (49.03 PPL, −4.86 vs float 53.90). Cycle 292 block 1 rank-1 retrain (50.08 PPL, −3.82 vs float). Float spirals/moons accuracy from this cycle's training (item 305 used full set; we use held-out).
