# Cycle 027 Plan — Centroid and Jacobian Affinity as Merge Signals

## Script
`experiments/centroid_jacobian_affinity_gpt2.py`

## Output
`autoresearch/results/cycle_027_affinity_signals.json`
`autoresearch/results/cycle_027_affinity_signals.png`

## Pre-flight
- P1: GPT-2 loads via `load_nanochat_teacher(TeacherLoadSpec(source="hf_pretrained", hf_model_name="gpt2"))`
- P2: Shakespeare corpus at `data/shakespeare.txt` — confirmed used in cycles 020–026
- P3: `.venv/bin/python` exists; run with `MPLCONFIGDIR=.mplconfig`
- P4: `autoresearch/results/` directory exists

## Algorithm

### Setup
- Load GPT-2, eval mode, no grad
- Build probe corpus: seq_len=512, n_sequences=2 (→ 1024 tokens)
- Run forward pass with hooks to collect FFN inputs and hidden activations per block

### Per-block (12 blocks)
For block b with W_up (3072×768), W_down (768×3072), bias_up (3072,):

**Step 1 — Collect activations**
Hook into `mlp.c_fc` (pre-GELU) to get z ∈ ℝ^{T×3072} and pre-MLP hidden state h ∈ ℝ^{T×768}.
Compute GELU activations: a = gelu(z), shape T×3072.

**Step 2 — Sample 200 channel pairs**
Sample 200 random pairs (i, j) with i < j, i,j ∈ [0, 3072).

**Step 3 — Compute signals**

a. Weight cosine:
   `cos_w[i,j] = dot(W_down[:,i], W_down[:,j]) / (||W_down[:,i]|| * ||W_down[:,j]||)`
   (W_down columns are the down-projection weights for each channel)

b. Centroid similarity:
   `mu_i = mean(a[:, i])` scalar, but use vector over tokens: `mu_i = a.mean(0)[i]`
   Actually: centroid = mean activation *pattern* — compute `c_i = a[:,i]` ∈ ℝ^T (activation
   time-series), then `cos_c[i,j] = dot(c_i, c_j) / (||c_i|| * ||c_j||)`.

c. Jacobian-spectrum similarity:
   J_block = W_down @ diag(gelu'(z)) @ W_up, shape T×768×768 is too large.
   Instead, use per-channel contribution: for channel i, contribution vector is
   `v_i(t) = W_down[:, i] * gelu'(z[t, i])` ∈ ℝ^{768} (what channel i writes to output).
   Aggregate over tokens: `V_i = (1/T) * sum_t v_i(t)` ∈ ℝ^{768}.
   `cos_j[i,j] = dot(V_i, V_j) / (||V_i|| * ||V_j||)`.

**Step 4 — Measure merge cost**
For each pair (i, j), compute merge cost:
- Weight mean: w_merge = (W_down[:,i] + W_down[:,j]) / 2, bias_up_merge = (bias_up[i] + bias_up[j]) / 2
- Activation merge: a_merge = (a[:,i] + a[:,j]) / 2 (equivalent to merged channel activation)
- Block output baseline: `out = a @ W_down.T` (simplified — W_down is (768,3072), so `out = a @ W_down.T`)
  Wait: GPT-2 MLP: `h = W_down(gelu(W_up(x)))` where W_up: 768→3072, W_down: 3072→768.
  In nn.Linear terms: `c_proj.weight` is (768, 3072), so `out = gelu(z) @ c_proj.weight.T + c_proj.bias`.
- Merged output diff: replace columns i,j of W_down with w_merge (zero one, double other):
  `delta_out = (a[:,j] - a[:,i]).unsqueeze(1) * (W_down[:,j] - W_down[:,i]).unsqueeze(0) / 2`...
  Simpler: compute `out_orig = a @ W_down.T`, then build `a_mod` = copy of a with col i set to
  `(a[:,i]+a[:,j])/2`, col j set to zero, `out_mod = a_mod @ W_down.T + a_mod_bias_correction`.
  Merge cost = `||out_orig - out_mod||_F^2 / T`.

**Step 5 — Correlate**
Spearman(signal, merge_cost) for each signal. Negative correlation = high similarity predicts low cost.
Use |ρ| or sign-flip: report ρ(−signal, merge_cost).

### Output
JSON: per-block dict with spearman_weight_cos, spearman_centroid, spearman_jacobian, n_pairs.
Plot: 3×12 grid showing ρ per block per signal, plus summary bar chart.

## Verification
- Check merge cost is always ≥ 0
- Sanity: weight_cos should give ρ > 0 (higher similarity → lower cost) — flag if not

## Runtime estimate
~5–10 min on CPU (no full Jacobian matrix, just per-channel vectors)
