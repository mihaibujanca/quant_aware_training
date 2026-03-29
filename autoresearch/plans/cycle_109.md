# Cycle 109 Plan: Negative Leak Decomposition — GPT-2 FFN

**Item:** 120 | **Script:** `experiments/negative_leak_decomposition_gpt2.py` (new)

## Pre-flights

**P1–P7:** Unchanged from round 1 (null hypothesis, criteria, novelty, confounds, cost, data, abort).

## Implementation

1. **Load model and data.** GPT-2 (124M), 4096 OWT tokens. No quantization.

2. **Hook all 12 FFN blocks.** Capture per block i:
   - `z_i`: pre-GELU activation, shape `(T, 3072)`
   - `W_out_i = model.transformer.h[i].mlp.c_proj.weight.T`  — Conv1D stores `(3072, 768)`, transpose to `(768, 3072)` **[I1 fix]**
   - `b_out_i = model.transformer.h[i].mlp.c_proj.bias`  — shape `(768,)` **[A2 fix]**
   - Use `F.gelu(z, approximate='tanh')` everywhere **[A1 fix]**

3. **Per-channel stats.** For each block i, channel j (j=0..3071):
   - `neg_mask = (z_i[:, j] < 0)`. Skip if `neg_mask.sum() < 50`.
   - `gelu_neg_j = F.gelu(z_i[neg_mask, j], approximate='tanh')`
   - `μ_leak_j = gelu_neg_j.mean()`, `var_leak_j = gelu_neg_j.var()`

4. **Three-way per-token decomposition [I2 fix].** For each token t:
   - `N(t) = {j : z_j(t) < 0}` — negative channel set for this token
   - `pos(t) = Σ_{j ∉ N(t)} W_out[:,j] · GELU(z_j(t))` — positive-regime contribution
   - `b_leak(t) = Σ_{j ∈ N(t)} W_out[:,j] · μ_leak_j` — bias component (token-dependent channel set, global per-channel mean)
   - `s_leak(t) = Σ_{j ∈ N(t)} W_out[:,j] · (GELU(z_j(t)) - μ_leak_j)` — signal component
   - `block_out(t) = pos(t) + b_leak(t) + s_leak(t) + b_out` — closed decomposition **[A2]**
   - Verify: `||block_out(t) - hooked_output(t)|| / ||hooked_output(t)|| < 1e-5`

5. **Per-block ratios.**
   - `b_ratio_i = mean_t(||b_leak(t)||) / mean_t(||block_out(t)||)`
   - `s_ratio_i = mean_t(||s_leak(t)||) / mean_t(||block_out(t)||)`
   - `b_energy_i = mean_t(||b_leak(t)||²) / mean_t(||b_leak(t) + s_leak(t)||²)`

6. **Top-20 variance channels.** Per block, rank by `var_leak_j`. Decode 5 tokens with largest `|GELU(z_j)|` in negative regime.

7. **Verdict.**
   ```
   if sum(e > 0.70 for e in b_energy) >= 8:  verdict = "CONFIRMED"
   elif sum(e < 0.30 for e in b_energy) >= 8: verdict = "REFUTED"
   else: verdict = "INCONCLUSIVE"
   ```

8. **Output.** JSON `autoresearch/results/cycle_109_negative_leak.json`, plot `cycle_109_negative_leak.png` (12-block bars: b_ratio, s_ratio, b_energy). Agenda update + commit.
