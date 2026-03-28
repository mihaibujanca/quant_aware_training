# Task
Item 92: Retained-spectrum denoising via analytic projection — GPT-2 FFN. After topk+lstsq compression of blocks [7,5,4,1] at 6x, compute per-block error on 4096 probe tokens. Project error onto the retained SVD basis of block output covariance. Re-solve lstsq for W_down targeting corrected outputs. Compare single-step analytic correction vs 500-step Adam polish on stacked PPL.

# Hypothesis
- Confirmed if: analytic correction achieves <= +10% PPL (within 2x of Adam's +4.9%).
- Refuted if: analytic correction gives > +25% PPL or diverges.
- Inconclusive if: helps (+10-25%) but doesn't match Adam closely.

# Pre-flight
- P1 (data): Shakespeare at `data/shakespeare.txt` (used in cycles 059/061/072).
- P2 (deps): `aleph/lm_geometric_compression` provides compression pipeline. Cycle 059/061 scripts demonstrate Adam polish on stacked blocks. All validated.
- P3 (ground truth): Unpolished stacked PPL = +30.2%, Adam 500-step = +4.9% (cycle 061). Baseline PPL ~57.65.
- P4 (compute): Compression + lstsq + SVD of output covariance + re-lstsq. All analytic, no gradient loop. ~5min on CPU.
- P5 (cross-block caveat): Analytic correction is per-block-local. It corrects each block's W_down independently. Cross-block error propagation (stacking amplifies 1.8-2.6x per cycle 072) is NOT addressed by single-pass correction. This is the likely failure mode.

# Steps
Script: `experiments/retained_spectrum_denoising_gpt2.py`

1. **Setup.** Load GPT-2 via `load_nanochat_teacher`. Shakespeare probe corpus, seq_len=128, 4096 tokens (32 seqs for probe_train, 32 for eval). Seeds [42, 123, 7]. Blocks = [7, 5, 4, 1]. HIDDEN_FRACTION = 1/6.

2. **Baseline compression (topk+lstsq).** For each seed: probe FFN geometry (`probe_ffn_geometry`), derive compression (`derive_ffn_compression` with selection_mode="topk"), swap into model (`swap_compressed_ffns`). Evaluate stacked PPL → this is the "unpolished" baseline (~+30%).

3. **Capture per-block errors.** On the stacked compressed model, hook each compressed block's MLP output AND the float model's corresponding MLP output on the same 4096 probe tokens. Per block b: `e_b = y_comp_b - y_float_b` (shape [N, 768]). Also capture `h_comp_b` (the compressed hidden activations = input to c_proj, shape [N, k_b]) for re-solving lstsq.

4. **Output covariance SVD.** Per block b: compute SVD of the float model's MLP output covariance. `Y_float_b` is [N, 768]. Center it: `Y_c = Y_float_b - Y_float_b.mean(0)`. SVD: `U, S, Vh = torch.linalg.svd(Y_c, full_matrices=False)`. The "retained" basis is `Vh[:r_b, :]` where r_b = rank capturing 99% of output variance (or use cycle 072's rank_99 values ~481-640). Store `V_retained = Vh[:r_b, :].T` (shape [768, r_b]).

5. **Project error and compute corrected targets.** Per block b:
   - `e_retained = e_b @ V_retained @ V_retained.T` — error projected onto retained output subspace.
   - `y_corrected = y_float_b - e_retained` — corrected target (item formula).
   - NOTE: This target says "when re-solving for W_down, aim for float output MINUS the retained-spectrum error." The idea: since lstsq already minimized overall error but left residual in the retained subspace, re-solving against corrected targets that account for this systematic bias should reduce it.

6. **Re-solve lstsq for W_down.** Per block b: `h_comp_b` is [N, k_b] (compressed hidden activations). Solve `W_down_new = argmin ||W_down @ h_comp - y_corrected||_F` via `torch.linalg.lstsq`. Replace the block's c_proj weight (and bias if present) with the new solution.

7. **Evaluate corrected stacked PPL.** Swap the corrected W_down values into the stacked model. Evaluate PPL on eval set. This is the "analytic correction" result.

8. **Adam polish comparison.** On a separate copy of the stacked model (from step 2, before analytic correction), run 500 Adam steps on W_down columns at lr=1e-4 on a SEPARATE holdout split (matching cycle 061 protocol — polish data must be disjoint from lstsq probe and eval). Evaluate PPL on eval set. This is the "Adam 500-step" reference.

9. **Also test: iterative analytic correction.** Repeat steps 3-6 two more times (total 3 passes). Each pass re-computes errors on the updated model and re-projects. If cross-block error is the bottleneck, iteration should help. Report PPL after each pass.

10. **Save results.** JSON: `autoresearch/results/cycle_080_analytic_projection.json`. Per seed: baseline_ppl, unpolished_ppl, analytic_1pass_ppl, analytic_2pass_ppl, analytic_3pass_ppl, adam_500_ppl. Aggregated: means and stds, delta_pct for each. Plot: bar chart comparing methods, save to `autoresearch/results/cycle_080_analytic_projection.png`.

11. **Verdict.** Compute mean delta_pct for analytic 1-pass (and best iterative pass). Compare vs thresholds.

Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/retained_spectrum_denoising_gpt2.py`

# Validation
- Unpolished baseline should be ~+30% PPL (matching cycle 061). If far off, probe corpus or compression config differs — stop.
- Adam 500-step should be ~+4.9%. If > +10%, something is wrong with the polish implementation.
- Analytic correction must not INCREASE PPL vs unpolished (if it does, the projection formula is wrong — check signs).
- Energy check: `||e_retained||^2 / ||e||^2` should be 0.77-0.91 per cycle 072 findings.

# Artifacts
- `experiments/retained_spectrum_denoising_gpt2.py`
- `autoresearch/results/cycle_080_analytic_projection.json`
- `autoresearch/results/cycle_080_analytic_projection.png`

# Fallbacks
- If analytic correction increases PPL: flip sign to `y_corrected = y_float + e_retained` and re-test. The formula interpretation may be wrong.
- If 1-pass is weak but iteration helps: report both. The item asks for single-step comparison but iterative is informative.
- If Adam baseline doesn't match cycle 061: use cycle 061's numbers as reference, note discrepancy.
