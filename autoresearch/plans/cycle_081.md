# Task
Item 85: Analytic stacking correction via Jacobian chain rule -- GPT-2 FFN. After topk+lstsq compression of blocks [7,5,4,1] at 6x, compute a single Newton step on each block's W_down using the full stacked model loss. Compare: (a) unpolished, (b) single Newton step, (c) 500 Adam steps.

# Hypothesis
- Confirmed if: single Newton step achieves ≤ +8% PPL (within 2x of Adam's +4.9%), making the pipeline fully analytic.
- Refuted if: Newton step diverges or gives > +20% PPL (landscape is too nonlinear for single-step correction).
- Inconclusive if: Newton step helps (+15-20%) but doesn't match Adam (curvature is significant, needs damping/iteration).

# Pre-flight Results
- P1: Not duplicate. Past experiments: per-block lstsq (080, vacuous), Adam polish (061, iterative), spectral decomp (072). None attempt single Newton step with full-model Jacobian.
- P2: Dependencies satisfied. Cycle 061 (Adam baseline) and 080 (lstsq no-op) both complete.
- P3: Concrete -- single script, 3 conditions, clear thresholds.
- P4: No program.md changes needed.
- P5 (null baseline): Unpolished = +30.2% PPL. A random W_down perturbation of matched norm would give ~+30% or worse. Adam 500 = +4.9% is the strong baseline. Steepest descent (gradient only, no curvature) is the intermediate null.
- P6 (methods): Gauss-Newton is appropriate for cross-entropy loss (PSD by construction via J^T J on logits). GPT-2 uses GELU -- smooth, so Jacobian is well-defined everywhere. No ReLU discontinuity issues.
- P7 (power): 200 eval sequences x 128 tokens = 25600 tokens. PPL differences of interest are 5-25pp on base ~57. 3 seeds provide variance estimates. Adequate for the effect sizes expected.

# Steps
Script: `experiments/newton_correction_stacked_gpt2.py`

1. **Setup.** Load GPT-2, OWT corpus (OWT_PATH), seq_len=128. probe_train=200 seqs (lstsq), holdout=200 seqs (Newton/Adam), eval=200 seqs. Seeds [42,123,7]. Blocks [7,5,4,1]. HIDDEN_FRACTION=1/6. batch_size=8 (halve to 4 on OOM, see step 3 guard).

2. **Baseline + compression.** Per seed: topk channel selection, lstsq refit, swap compressed FFNs. Eval stacked PPL. Must be ~+30% to match cycle 061. If <+20% or >+40%, abort.

3. **Gauss-Newton correction (primary method), applied block-by-block in order [7,5,4,1].** For block b:
   a. Freeze all params. Unfreeze W_down_b (shape [768, k_b], ~98K params).
   b. Compute gradient g = dL/d(W_down_b) on holdout data (batched, batch_size from step 1).
   c. Define Gauss-Newton Hv product (PSD by construction):
      ```
      # Forward: logits = model(tokens)  [no softmax]
      # Jv:  jvp_out = jvp(logits, w, v)  -- forward-mode tangent
      # J^T(Jv): GNHv = vjp(logits, w, jvp_out)  -- backward-mode cotangent
      ```
      Accumulate GNHv over mini-batches (batch_size). Call `torch.cuda.empty_cache()` between blocks.
   d. **OOM guard**: wrap Hv computation in try/except RuntimeError. On OOM: halve batch_size, retry. Log actual batch_size used.
   e. Solve `(J^T J) dw = -g` via CG (max_iter=50, tol=1e-4). CG only calls GNHv, never materializes the matrix.
   f. Apply: `W_down_b.data += dw.reshape(W_down_b.shape)`. Re-freeze block b.
   g. **Log per-block PPL on eval set after each block correction.** This tracks whether earlier corrections go stale as later blocks are corrected.

4. **Evaluate Newton-corrected stacked PPL.** Eval set. This is the primary result.

5. **Second pass check.** After all 4 blocks corrected, re-evaluate gradient norm for block 7 (first-corrected). If grad_norm > 0.1 * original_grad_norm, run a second pass over all blocks and report PPL delta. If second pass changes PPL by >1pp, note block-order sensitivity.

6. **Fallback: true Hessian + Tikhonov damping.** Only if Gauss-Newton CG fails (NaN, divergence, or PPL worse than unpolished). Use `Hv = autograd.grad(g @ v, w, create_graph=True) + lambda * v` with lambda = 0.01 * ||g|| / n_params. This is the backup, not the primary method.

7. **Line-search damping.** If raw Newton step worsens PPL vs unpolished, try alpha in [0.1, 0.3, 0.5, 0.7] and report best.

8. **Steepest descent baseline.** On fresh compressed copy: per-block `W_down_b -= lr * g` with optimal lr = ||g||^2 / (g^T GNHv(g)). This isolates curvature benefit of Newton vs gradient-only.

9. **Adam 500-step baseline.** Fresh compressed copy: 500 Adam steps on all W_down jointly, lr=1e-4, holdout data. Must match cycle 061 (~+4.9%). If >+10%, implementation error.

10. **Save results.** JSON: `autoresearch/results/cycle_081_newton_correction.json`. Per seed: baseline_ppl, unpolished_ppl, per_block_ppl_sequence (4 entries after each block correction), newton_ppl, newton_second_pass_ppl (if run), steepest_descent_ppl, adam_500_ppl. Plot: `autoresearch/results/cycle_081_newton_correction.png`.

11. **Bookkeeping.** Move item 85 from NEXT to PAST EXPERIMENTS in `autoresearch/agenda.md` with commit hash and artifact paths. Commit: `[autoresearch] cycle 081: analytic Newton correction via Gauss-Newton -- GPT-2`.

# Validation
- Unpolished baseline must be ~+30% PPL (cycle 061 match). Abort if outside [+20%, +40%].
- Adam 500-step must be ~+4.9%. If >+10%, implementation error -- do not trust Newton results.
- CG must converge (residual < tol) in at least 2/3 seeds. If 0/3, method is broken.
- Gradient norms should be O(0.01-1), not O(1e6). If huge, check loss scaling.
- Per-block PPL sequence must be monotonically improving (or at worst flat). If any block correction worsens PPL by >2pp, flag block-order issue.

# Artifacts
- `experiments/newton_correction_stacked_gpt2.py`
- `autoresearch/results/cycle_081_newton_correction.json`
- `autoresearch/results/cycle_081_newton_correction.png`

# Fallbacks
- GN CG diverges -> true Hessian + Tikhonov damping (step 6).
- Newton step worsens PPL -> line-search damping (step 7).
- All Newton variants fail -> report steepest descent result (step 8). This still tests whether full-model gradient direction beats per-block lstsq.
- OOM on create_graph -> halve batch_size (step 3d). If still OOM at batch_size=1, switch to CPU for Hv products (slower but tractable for GPT-2 small).
