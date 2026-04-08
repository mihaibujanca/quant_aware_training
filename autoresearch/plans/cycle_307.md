# Cycle 307 Plan — Item 317: Higher-dim input eff_rank census (50D)

## Item 317 verbatim Hypothesis
On a 50D-input 32×12 MLP trained to ≥95% accuracy on a linearly-separable-in-50D task, init mean eff_rank ≥ 20 at layers 1-6 (clean high-rank precondition), and POST-training mean eff_rank ≤ 5 at 10/12 layers — reproducing the cycle 278 GPT-2 bathtub qualitatively on a controlled MLP. This would be the first time the "collapse to rank-2" framing is validated on an MLP.

## Dependencies
- `aleph.models.MLP` (input_size, hidden_size, output_size, depth, activation_fn)
- `experiments/spirals_moons_eff_rank_cycle295.py` (template: `collect_post_relu`, `eff_rank_from_Y`, `train_until_acc`)
- sklearn ≥1.0 for `make_classification`
- Cycle 295 post_eff per layer (2D baseline): spirals `[3.14, 2.86, 2.11, 2.09, 2.15, 1.94, 1.99, 2.07, 2.05, 2.03, 1.76, 1.74]`

## Approach
1. New script `experiments/higher_dim_eff_rank_cycle307.py` mirroring cycle 295 structure (dataset factory → build_mlp → collect_post_relu → train_until_acc → eff_rank_from_Y → aggregate → verdict).
2. Dataset: `sklearn.datasets.make_classification(n_samples=2000, n_features=50, n_informative=10, n_classes=2, random_state=seed)`. Standardise (per-feature z-score on the drawn sample; no leakage since no held-out split).
3. Architecture: `MLP(input_size=50, hidden_size=32, output_size=1, depth=12, activation_fn=nn.ReLU)`.
4. Training: full-batch Adam lr=1e-3, BCEWithLogitsLoss, stop when train_acc ≥ 0.90 (item brief relaxes to 0.90 vs cycle 295's 0.95), max_epochs=20000.
5. Per-layer captures at init AND post-training via `collect_post_relu`. eff_rank via `eff_rank_from_Y` (spectral entropy on Y^T Y / N).
6. Additionally record eff_rank trajectory at epochs {0, 50, 200, 500, 2000} — per cycle 297 clause in the agenda text — to distinguish monotone collapse vs lift.
7. Seeds {42, 137, 2024}. Per-seed records + per-layer mean/std across seeds, per item-305/306/317 convention.
8. Write JSON, summary, manifest, exec_result, run log, agenda/STATUS/TSV updates, audit. CPU only; wall < 2 min estimated (depth-12 32-wide × 3 seeds, linear-separable task, small N).

## Files
- New: `experiments/higher_dim_eff_rank_cycle307.py`, `autoresearch/results/cycle_307_eff_rank_50d.json`, `autoresearch/results/cycle_307_summary.md`, `autoresearch/results/cycle_307_manifest.txt`, `autoresearch/results/cycle_307_exec_result.txt`, `autoresearch/audits/cycle_307.md`, `autoresearch/results/cycle_307_audit.txt`, `autoresearch/logs/cycle_307_run.log`.
- Updated: `autoresearch/agenda.md`, `autoresearch/logs/experiments.tsv`, `autoresearch/STATUS.md`.

## Config
`HIDDEN=32, DEPTH=12, N_SAMPLES=2000, N_FEATURES=50, N_INFORMATIVE=10, N_CLASSES=2, LR=1e-3, MAX_EPOCHS=20000, TRAIN_ACC_TARGET=0.90, SEEDS=[42,137,2024], TRAJECTORY_EPOCHS=[0,50,200,500,2000]`, CPU, full-batch BCEWithLogits.

## Verification
- `len(collect_post_relu(model, X)) == DEPTH == 12`.
- Each activation tensor has shape `(N_SAMPLES, HIDDEN) == (2000, 32)`.
- `eff_rank` is in `[1.0, HIDDEN]`; at init with 50D input and 32 hidden units, expect init eff_rank ~ min(50, 32) = 32 at layer 0 (decays through layers under random ReLU).
- Train-acc ≥ 0.90 per seed (asserted); otherwise mark `converged=False`.
- Sanity cross-check: cycle 295 spirals init eff_rank `[2.99, 2.71, 1.87, 1.19, 1.06, 1.01, 1.00, ...]` is the 2D baseline to beat at init.

## Verdict logic (item 317 verbatim)
- **CONFIRMED** iff (a) init mean eff_rank ≥ 15 at layers 1-6 across all 3 seeds (6 layers × 3 seeds = all 18 must pass) AND (b) post-training mean eff_rank ≤ 5 at ≥8 of 12 layers across all 3 seeds AND (c) trajectory is MONOTONE DECREASING at ≥6 of 12 layers (layer monotone iff across the 5 trajectory epochs the per-epoch mean eff_rank is non-increasing).
- **REFUTED** iff init mean eff_rank ≥ 15 holds at layers 1-6 (precondition present) but post-training eff_rank ≤ 5 at <4 of 12 layers.
- **INCONCLUSIVE** otherwise.

# APPROVE
