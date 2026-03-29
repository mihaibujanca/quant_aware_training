# Task
Item 93: Gauss-Newton correction retry with true J^T J — GPT-2 FFN

# Hypothesis
A single Gauss-Newton step using the true PSD approximation H_gn = J^T J (where J = d(per-token-losses)/d(vec(W_down)), shape [n_tokens, n_params]) with proper CG convergence (100+ iters) and meaningful Tikhonov damping can recover most of the stacked compression error.

Confirmed if: single GN step achieves ≤ +12% PPL (within 2× of 500-step Adam's +4.9%).
Refuted if: GN step diverges or gives > +25% PPL despite correct PSD Hessian and ≥100 CG iters.
Inconclusive if: +12–25% PPL (helps but doesn't match Adam).

# Steps

1. **Create `experiments/gauss_newton_true_gpt2.py`** (new file):
   - **Jacobian target: loss-Jacobian.** J = d(per-token CE losses)/d(vec(W_down)), shape [n_tokens, n_params]. With HV_SEQS=64 × 128 positions = 8192 tokens and 98K params, this is underdetermined (8K < 98K). The effective rank of J^T J is at most 8K, so CG residual will plateau before 100 iters — this is expected, not a bug. Tikhonov damping regularizes the null space.
   - **Hv products via jvp+vjp** (never materialize J):
     - Forward: `Jv = jvp(loss_per_token, W_down, v)` — one forward pass with tangent
     - Backward: `JtJv = vjp(loss_per_token, W_down)(Jv)` — one backward pass
     - Result is J^T J @ v, PSD by construction.
   - Process in mini-batches of 4 sequences, accumulate Hv products.
   - Fallback: if `torch.func` unavailable, use `autograd.grad` with `create_graph=True`.

2. **Hyperparameters**:
   - `CG_MAX_ITER = 100`, `CG_TOL = 1e-5`
   - Damping sweep: lambda in **[1e-4, 1e-3, 1e-2, 1e-1]** for block 7 seed 42 only; pick best, use for full run.
   - `HV_SEQS = 64`, `BATCH_SIZE = 4`
   - Blocks [7,5,4,1] at 6×, seeds [42, 123, 7].

3. **Baselines** (same run):
   - Unpolished (no correction), steepest descent (Rayleigh quotient LR), Adam 500 steps.
   - Log CG residual norm at iterations [10, 25, 50, 100]. Residual plateau is expected in the underdetermined regime — only monotonic increase indicates a bug.

4. **OOM fallback**: try/except per block. On OOM: empty_cache, halve HV_SEQS to 32. If still OOM, fall back to steepest descent for that block and log it.

5. **Artifacts**: `autoresearch/results/cycle_082_gauss_newton_true.json` (per-block CG curves, lambda used, Hv method tag "jvp_vjp" or "autograd"), `cycle_082_gauss_newton_true.png`.

6. **Validation checks**:
   - CG residual must not increase (PSD guarantee). Increase → abort, report Hv bug.
   - Adam 500 must reproduce ~+5%. Unpolished must be ~+25-35%. Outside range → setup bug.

7. **Agenda update**: move item 93 to PAST EXPERIMENTS with outcome.

8. **Commit**: `[autoresearch] cycle 082: true Gauss-Newton correction — GPT-2 FFN`

# Artifacts
- `experiments/gauss_newton_true_gpt2.py`
- `autoresearch/results/cycle_082_gauss_newton_true.json`
- `autoresearch/results/cycle_082_gauss_newton_true.png`
