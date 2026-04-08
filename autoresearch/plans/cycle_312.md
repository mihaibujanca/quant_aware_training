# Cycle 312 Plan — Item 322

## Item 322 (verbatim Hypothesis)
cycle 307 attributed the L4+ init eff_rank collapse (1.05 → 1.00) on 50D input to "hidden_width=32 is the binding constraint past layer 0; Hanin-Rolnick N^d activation-pattern bottleneck applies once the layer width is below the data manifold dim". A wider net (hidden=128 = 2.56× input_dim, vs cycle 307's 32 = 0.64× input_dim) should preserve init eff_rank well past layer 0 if the bottleneck story is correct. Test whether the GPT-2 cycle 278 bathtub (b1=1.94, b2=1.05) finally reproduces in any feasible MLP setting OR whether ReLU MLPs **never** collapse trained eff_rank in this regime. **Setup**: aleph.models.MLP(input_size=50, hidden_size=128, output_size=1, depth=12, ReLU); sklearn.make_classification(n_samples=2000, n_features=50, n_informative=10, n_classes=2, n_redundant=0, n_repeated=0, n_clusters_per_class=1, flip_y=0.0, class_sep=1.0, random_state=seed) + per-feature z-score; full-batch BCEWithLogits; Adam LR=1e-3 max_epochs=20000 train_acc_target=0.90; SEEDS={42, 137, 2024}; trajectory at [0, 50, 200, 500, 2000]; CPU. Compute eff_rank = (sum sigma)^2 / sum sigma^2 of activations (2000, 128) at each of 12 hidden layers, both at init and post-train, all 3 seeds. **Verbatim conditions**: post mean eff_rank ≤ 5 at ≥8/12 layers across all 3 seeds AND trajectory monotone-decreasing at ≥6/12 layers → CONFIRMED; init mean eff_rank ≥ 60 at all 12 layers AND post mean eff_rank ≥ 30 at all 12 layers → REFUTED; else INCONCLUSIVE.

## Dependencies
- `experiments/higher_dim_eff_rank_cycle307.py` — direct template. Cycle 307 result (INCONCLUSIVE): init [15.70, 5.93, 2.96, 1.35, 1.05, 1.01, 1.00, 1.00, ...]; post [13.48, 2.37, 1.75, 1.74, ...]; hidden=32. Cycle 278 GPT-2 bathtub: b1=1.94, b2=1.05.
- Note definition change: item 322 uses stable rank `(sum sigma)^2 / sum sigma^2` of activations directly (not spectral entropy of the Gram eigenvalues like cycle 307). Stable rank on singular values of Y (2000, 128).

## Approach
1. New script `experiments/higher_width_mlp_cycle312.py`. Copy cycle 307 verbatim except: HIDDEN=128 (was 32) and `eff_rank_from_Y` uses stable rank on SVD of Y ((sum s)^2 / sum s^2) per item 322. All other config (depth, seeds, lr, max_epochs, target_acc, trajectory epochs, dataset params) identical.
2. Per seed: build_mlp(50, 128, 1, depth=12, ReLU); capture init post-ReLU activations (12 layers, shape (2000, 128)); full-batch Adam BCEWithLogits until train_acc ≥ 0.90; capture trajectory at {0, 50, 200, 500, 2000}; capture post-train activations. Compute stable rank at each layer at each snapshot.
3. Aggregate: per-layer mean/std across 3 seeds for init and post; per-epoch mean trajectory across seeds.
4. Apply verdict per item 322 verbatim: (a) post mean ≤ 5 at ≥8/12 AND (b) trajectory monotone non-increasing at ≥6/12 (per-layer across epoch series) → CONFIRMED; (c) init mean ≥ 60 at ALL 12 AND post mean ≥ 30 at ALL 12 → REFUTED; else INCONCLUSIVE.
5. Cross-compare to cycle 307 (HIDDEN=32) per-layer init/post — note cycle 307 used entropy not stable rank so direct numeric comparison is rough; report both.

## Files
- `experiments/higher_width_mlp_cycle312.py` (new)
- `autoresearch/results/cycle_312_eff_rank_50d_w128.json`, `_summary.md`, `_manifest.txt`, `_exec_result.txt`
- `autoresearch/logs/cycle_312_run.log`, `autoresearch/audits/cycle_312.md`, `autoresearch/results/cycle_312_audit.txt`

## Config
SEEDS=[42,137,2024]; HIDDEN=128; DEPTH=12; N_SAMPLES=2000; N_FEATURES=50; N_INFORMATIVE=10; LR=1e-3; MAX_EPOCHS=20000; TRAIN_ACC_TARGET=0.90; TRAJECTORY_EPOCHS=[0,50,200,500,2000]; device=CPU. Activations shape per layer = (2000, 128). Trainable params ~= 50*128 + 11*(128*128) + 128+1 ≈ 186,881 per net.

## Verification
- All 3 seeds reach train_acc ≥ 0.90 within MAX_EPOCHS (make_classification n_informative=10 flip_y=0 class_sep=1.0 is trivially separable; cycle 307 hit 1.000 at epoch 50).
- `len(model.layers) == 2*DEPTH + 1` asserted; 12 post-ReLU activations captured.
- Stable rank ∈ [1.0, 128.0] per layer per snapshot.
- eff_rank at init for random Gaussian ReLU: layer 0 should be ≲ min(input rank, hidden)/factor; layers >0 bounded by hidden=128 and ReLU activation-pattern constraint.

## Verdict logic (item 322 verbatim)
CONFIRMED iff n_layers_post_le5 ≥ 8 (across all 3 seeds) AND n_layers_monotone ≥ 6. REFUTED iff init_mean[L] ≥ 60 ∀L AND post_mean[L] ≥ 30 ∀L. INCONCLUSIVE otherwise.

## Baselines to record
Cycle 307 hidden=32 post mean (entropy): [13.475, 2.370, 1.745, 1.741, 1.767, 1.767, 1.712, 1.677, 1.551, 1.524, 1.547, 1.743]. Cycle 278 GPT-2 c_proj output eff_rank (subset): b0=71.93, b1=1.94, b2=1.05, b3=24.29, b4=49.04, b5=179.19, b9=195.5, b11=22.49. Random-init expectation at width 128: stable rank O(128) pre-ReLU, reduced by ReLU.

# APPROVE
