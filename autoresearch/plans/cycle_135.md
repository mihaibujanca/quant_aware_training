# Cycle 135 Plan: Curved Inference Census (Item 145)

## Task
Compute salience and curvature profiles of GPT-2's residual stream under the pullback metric G = U^T U, then correlate curvature with single-block swap PPL damage at 6x compression.

## Hypothesis
If GPT-2's residual stream exhibits the same geometric structure as larger models (Manson 2025), salience and curvature should be anticorrelated (|rho| >= 0.7), and high-curvature blocks should be harder to compress (curvature-damage |rho| >= 0.5).

## Pre-flight Checks
- P1 (duplicate): no prior cycle computed pullback curvature on GPT-2.
- P2 (deps): none. Item 146 depends on THIS item.
- P3 (scope): 200 sequences x 3 seeds for trajectory collection (~2 min per seed). 12 single-block swap PPL evals at 6x (~15 min total). Single-cycle.
- P4 (program.md): no modifications needed.
- P5 (null baseline): for 11 curvature data points, Spearman |rho| >= 0.7 has p < 0.02. Meaningful if achieved.
- P6 (memory): residual trajectory is 13 x [1, 1024, 768] float32 = ~40 MB. Trivial.

## Steps

### Part A: Residual Stream Geometry (per seed)
1. Load GPT-2 (124M), tokenizer. Load 200 OWT sequences (1024 tokens each).
2. For each sequence batch (batched by 10 for memory):
   a. `trajectory = collect_residual_trajectory(model, input_ids)` — 13 tensors.
   b. `U = get_unembedding(model)` — [50257, 768].
   c. `S = compute_salience(trajectory, U, reduce="none")` — [12, batch, seq_len].
   d. `kappa = compute_curvature(trajectory, U, reduce="none")` — [11, batch, seq_len].
   e. Accumulate per-layer means and stds (online Welford).
3. After all sequences: compute mean salience profile [12] and mean curvature profile [11].
4. Compute Spearman(S[1:], kappa) — align indices (curvature is layers 1-11, salience 0-11; use S[1:12] to match kappa[0:11] which correspond to blocks 1-11).

### Part B: Single-Block Swap PPL at 6x (all 12 blocks)
5. For each block_id in range(12):
   a. Fresh model copy.
   b. Collect FFN activations (hidden, output) for the block on 200 OWT sequences.
   c. Topk selection: select top-512 channels by activation magnitude (6x = 512/3072).
   d. Lstsq refit: W_down_compressed via least-squares on selected channels.
   e. Inject compressed FFN, evaluate PPL on eval split.
   f. Record PPL delta = compressed_PPL - baseline_PPL.
6. Repeat for 3 seeds (different OWT subsets for calibration).

### Part C: Correlations and Report
7. Compute Spearman(kappa[0:11], PPL_delta[0:11]) — curvature vs damage for blocks 0-11.
   Note: kappa has 11 values (blocks 1-11); PPL_delta has 12 (blocks 0-11). For correlation, use PPL_delta[1:12] aligned with kappa.
8. Save results JSON: salience_profile, curvature_profile, ppl_deltas, correlations, per-seed values.
9. Generate 2-panel figure: (a) salience + curvature profiles, (b) curvature vs PPL-delta scatter.

## Validation
- Sanity: baseline PPL should be ~30 (GPT-2 on OWT). If outside [20, 50], tokenization or data is wrong.
- Sanity: salience should be positive everywhere (it's a norm).
- Sanity: curvature should be non-negative (it's a norm ratio).
- Sanity: at least blocks 1 and 4 should show measurable PPL delta at 6x (known from prior cycles).
- Cross-check: S_l can also be computed as `torch.linalg.norm((trajectory[l+1] - trajectory[l]) @ U.T, dim=-1).mean()`. Verify this matches `compute_salience` output for one batch.

## Criteria (verbatim from agenda)
- **Confirmed** if: salience-curvature anticorrelation |rho| >= 0.7 AND curvature-damage |rho| >= 0.5.
- **Refuted** if: |salience-curvature rho| < 0.3.
- **Inconclusive** if: salience-curvature |rho| in [0.3, 0.7), or curvature-damage |rho| < 0.3.

## Artifacts
- `experiments/curved_inference_census_cycle135.py`
- `autoresearch/results/cycle_135_curved_inference.json`
- `autoresearch/results/cycle_135_curved_inference.png`

## Fallbacks
- If `collect_residual_trajectory` hook captures wrong tensors (e.g., pre-residual-add): manually hook `model.transformer.h[i]` output and verify shape. The function already does this but check output[0] vs output tuple structure.
- If 12 single-block swap PPL evals are too slow (>30 min): run blocks [0, 1, 3, 5, 7, 9, 11] (7 blocks) and interpolate. Minimum viable: 7 points still gives meaningful Spearman.
- If curvature is near-zero at most layers (degenerate): report as INCONCLUSIVE with note that GPT-2 trajectory may be too shallow (12 layers) for meaningful curvature measurement.
