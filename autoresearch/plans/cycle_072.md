# Task
Item 83: Stacking error spectral decomposition — GPT-2 FFN. For k=4 blocks [7,5,4,1] at 6x (no polish), decompose per-block error onto mean-Jacobian singular vectors. Compare single-block vs stacked spectral profiles.

# Hypothesis
- **Confirmed if**: >=70% of stacking error energy is in discarded spectral directions AND stacking shifts error into retained spectrum vs single-block.
- **Refuted if**: error is uniformly distributed (no spectral structure) — random projection captures same fraction as Jacobian SVD.
- **Inconclusive if**: error is concentrated in discarded directions but stacking does NOT shift profile.

# Pre-flight
P1-P4 PASS. P5: Random-direction null — project error onto random orthonormal basis, expect ~r_b/768 fraction in any r_b-subset (uniform). P6: Conv1D transposition: `W_in = c_fc.weight.T` (3072,768), `W_out = c_proj.weight.T` (768,3072). P7: 512 tokens x 3 seeds x 4 blocks x 2 conditions = manageable on CPU.

# Steps
Script: `experiments/stacking_error_spectral_decomposition.py`

1. **Setup**: Load GPT-2 via `load_nanochat_teacher`. Shakespeare probe corpus, seq_len=128, 512 tokens. Seeds [42, 123, 7]. Blocks = [7, 5, 4, 1]. `HIDDEN_FRACTION = 1/6`.

2. **Compute mean Jacobian SVD per block**: For each block b, hook `mlp.c_fc` to get input x and pre-activation z per token. Form `J(x) = W_out @ diag(GELU'(z)) @ W_in` (768x768). Average over all tokens → J_bar_b. Full SVD of J_bar_b → `U_b, S_b, Vh_b`. Compute rank_99 FROM S_b directly: smallest r such that `sum(S_b[:r]^2) / sum(S_b^2) >= 0.99`. Do NOT use cycle 068 rank_99 values (those are per-token averages, SVD(E[J]) != E[SVD(J)]).

3. **Single-block error**: For each block b, compress it alone (topk+lstsq). Hook float and compressed models to capture FFN output on identical input tokens. `e_single_b = FFN_compressed(x) - FFN_float(x)`, shape (n_tokens, 768). This is block b's local compression error.

4. **Stacked (accumulated) error**: Compress ALL 4 blocks simultaneously. Hook each block's MLP on both models. `e_stacked_b = MLP_output_stacked(x_stacked) - MLP_output_float(x_float)` per block. NOTE: this is the total accumulated error at block b's output (upstream compression changes b's input), NOT block b's local compression error. This is the meaningful quantity for understanding stacking degradation: where does the accumulated error live in block b's spectral basis?

5. **Spectral projection**: For each error matrix e (n_tokens, 768) and U_b with rank_99 = r_b (from step 2):
   - `U_retained = U_b[:, :r_b]`, `U_discarded = U_b[:, r_b:]`
   - `frac_retained = ||e @ U_retained||^2 / ||e||^2`, `frac_discarded = 1 - frac_retained`
   - Cumulative energy: for k=1..768, `cum(k) = ||e @ U_b[:, :k]||^2 / ||e||^2`
   - **Null**: random orthonormal Q (QR of randn). Same projection. Expected frac ~ r_b/768.

6. **Plot**: 2x2 grid (one per block). Each: cumulative energy for single (blue), stacked (red), null (grey dashed). Vertical line at rank_99. Title: block id, r_b, frac_retained for both. Save `autoresearch/results/cycle_072_stacking_error_spectrum.png`.

7. **Save JSON**: `autoresearch/results/cycle_072_stacking_error_spectrum.json`. Per-block per-seed: frac_retained_{single,stacked,random}, total_error_norm_{single,stacked}, rank_99_from_J_bar, cumulative energy arrays (linearly spaced indices at k=1,16,32,...,768 — 49 points), outcome.

8. **Classify**: Is stacking error more retained-aligned than single-block error?

Run: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/stacking_error_spectral_decomposition.py`

# Validation
- Shape: J (768,768), U (768,768), error (n_tokens, 768).
- Energy conservation: frac_retained + frac_discarded in [0.999, 1.001].
- Null: frac_retained_random ~ r_b/768 (±5%).
- Cross-seed std of frac_retained < 0.10.

# Artifacts
- `experiments/stacking_error_spectral_decomposition.py`
- `autoresearch/results/cycle_072_stacking_error_spectrum.json`
- `autoresearch/results/cycle_072_stacking_error_spectrum.png`

# Fallbacks
- If mean Jacobian SVD yields rank_99 > 700 (near full rank), flag as degenerate and report raw cumulative curves without retained/discarded split.
- If energy conservation fails >1%, clamp GELU derivative inputs to [-10, 10].
