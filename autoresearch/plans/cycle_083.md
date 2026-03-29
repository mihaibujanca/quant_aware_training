# Task
GELU near-linearity census v2 for GPT-2 FFN (item 94). For each of 12 blocks × 3072 hidden channels, classify channels as dead, near-linear, or genuinely nonlinear using GELU derivative on 4096 mixed-corpus probe tokens. Corrects cycle 032's flawed post-GELU thresholding.

# Hypothesis
- **Confirmed** if: ≥10% dead OR ≥20% near-linear channels in ≥6/12 blocks.
- **Refuted** if: <5% dead AND <10% near-linear across all blocks.
- **Inconclusive** if: significant but uneven across blocks (<6/12 meet threshold).

# Pre-flight Results
- P1: Not duplicate — item 47/cycle 032 used post-GELU magnitude (ReLU-style), this uses GELU derivative.
- P2: No dependencies.
- P3: Single analysis cycle.
- P4: No program.md changes needed.
- P5: Null baseline — for Gaussian z ~ N(0,1), P(|GELU'(z)| < 0.01) ≈ 0.3%, P(|GELU'(z)-1| < 0.05) ≈ 28%. Report these alongside empirical results.
- P6: GPT-2 uses tanh-approx NewGELU: `0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))`. Use `F.gelu(z, approximate='tanh')` + autograd to get the exact derivative.
- P7: 4096 tokens (1024 each from shakespeare, wikitext2, OWT, C4) adequate for per-channel statistics (3072 channels, 4096 observations each).

# Steps
1. Create `experiments/gelu_nearlinearity_census_v2.py`.
2. Load GPT-2 via `load_nanochat_teacher(TeacherLoadSpec())`. Build 4 probe corpora (1024 tokens each from `data/shakespeare.txt`, `data/wikitext2_valid.txt`, `data/openwebtext_sample.txt`, `data/c4_sample.txt`) via 4× `ProbeCorpusSpec(text_path=..., probe_train_sequences=8, sequence_length=128)`. Concatenate probe_train tensors → (32, 128) = 4096 tokens.
3. For each block i (0–11), register forward hook on `model.transformer.h[i].mlp.act` to capture input (= pre-GELU z). Run forward pass in batches of 4 sequences, collect z tensors → shape (4096, 3072) per block.
4. Compute GELU derivative numerically: `z.requires_grad_(True); g = torch.nn.functional.gelu(z, approximate='tanh'); gelu_prime = torch.autograd.grad(g.sum(), z)[0]`. The `approximate='tanh'` matches GPT-2's NewGELU exactly.
5. Per channel j, per block: `dead_frac = (gelu_prime[:, j].abs() < 0.01).float().mean()`, `linear_frac = ((gelu_prime[:, j] - 1).abs() < 0.05).float().mean()`. Channel is "dead" if dead_frac > 0.95, "near-linear" if linear_frac > 0.80, else "nonlinear".
6. Compute Gaussian null baseline: sample z ~ N(0,1) shape (100000,), compute GELU' via autograd, report same dead/linear fractions.
7. Save JSON: `{block_i: {n_dead, n_linear, n_nonlinear, pct_dead, pct_linear, channel_details: [{j, dead_frac, linear_frac}]}}` + null baseline. Save to `autoresearch/results/cycle_083_gelu_census_v2.json`.
8. Plot 12-block stacked bar chart (dead/near-linear/nonlinear fractions) with null baseline line. Save to `autoresearch/results/cycle_083_gelu_census_v2.png`.
9. Evaluate confirmed/refuted/inconclusive per criteria above.

# Validation
- Sanity: GELU'(0) = 0.5, GELU'(3) ≈ 1.0, GELU'(-3) ≈ 0.004. Assert these before proceeding.
- Null baseline must show ~0% dead, ~28% near-linear (for N(0,1)). If empirical differs substantially from null, result is meaningful.
- Cross-check: sum of dead+linear+nonlinear = 3072 per block.

# Artifacts
- `experiments/gelu_nearlinearity_census_v2.py`
- `autoresearch/results/cycle_083_gelu_census_v2.json`
- `autoresearch/results/cycle_083_gelu_census_v2.png`
- Agenda update: mark item 94 with result, add cycle tag.
- Commit: `[autoresearch] cycle 083: GELU near-linearity census v2 — GPT-2`

# Fallbacks
- If `F.gelu(z, approximate='tanh')` autograd fails: compute derivative analytically for tanh-approx GELU: `gelu_tanh(x) = 0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))`, differentiate symbolically.
- If 4096 tokens OOM on CPU: reduce to 2048 (512 per source), note in results.
- If all blocks show <5% dead and <10% near-linear: result is REFUTED — GELU nonlinearity is fully utilized. Report and close.
