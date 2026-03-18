# Cycle 022 Plan — Attention Head Geometry Census (GPT-2)

## Pre-flight checks
- P1: `.venv/bin/python -c "import transformers, torch; print('ok')"` — verify env
- P2: GPT-2 Shakespeare probe corpus loads (same as cycle 009 pattern)
- P3: `model.transformer.h[0].attn.c_proj` exists and shape is [768, 768]
- P4: effective_rank formula: `torch.linalg.matrix_norm(A,'nuc') / torch.linalg.matrix_norm(A,2)`

## Script: `experiments/attn_head_census.py`

### Step 1 — Load model + corpus
Use same pattern as `experiments/jacobian_variance_atlas.py`:
- `load_nanochat_teacher` / GPT-2 from HuggingFace
- ProbeCorpusSpec with Shakespeare, 32 sequences × 128 tokens

### Step 2 — Register hooks on each attention layer
For each layer i in 0..11:
  Hook `model.transformer.h[i].attn` forward to capture:
  - `attn_weights`: shape [B, n_heads, seq_len, seq_len] (post-softmax)
  - `head_outputs`: V_head @ attn_weights^T for each head, shape [B, n_heads, seq_len, d_head]
  To get head outputs: inside hook, compute them from the raw V projections.
  Note: GPT-2 uses Conv1D. `c_attn` projects input to [B, T, 3*768].
  Split last dim into Q, K, V. Reshape to [B, T, n_heads, d_head], transpose.

### Step 3 — Collect over all probe sequences
Accumulate over sequences:
- `attn_weights_sum[i][h]`: sum of attn matrices [T, T], accumulate count for mean
- `head_outputs_all[i][h]`: list of [T, d_head] tensors (concatenate across sequences)

### Step 4 — Compute per-head metrics
For each head (i, h):
a. `A_mean = attn_weights_sum[i][h] / count` — shape [T, T]
   Effective rank = nuclear_norm(A_mean) / spectral_norm(A_mean)
b. Head output matrix H = concatenated [N_tokens, d_head]
   Variation = std(||H[t,:||_2 over t)
c. Within-layer CKA: for each pair (h1, h2) in same layer,
   CKA(H_h1, H_h2) using linear kernel: HSIC(X,Y) / sqrt(HSIC(X,X)*HSIC(Y,Y))
   HSIC(X,Y) = (1/n^2) * ||X^T Y_c||_F^2 where Y_c is mean-centered

### Step 5 — Prunability score
Normalize eff_rank and variation to [0,1] per layer.
Mean CKA of head h with all other heads in same layer = redundancy_h.
prunability[i][h] = (1 - eff_rank_norm) * (1 - variation_norm) * redundancy

### Step 6 — Intervention
Sort all 144 heads by prunability descending.
For K in [5, 10, 20, 30]:
  Copy model. For each pruned head (i,h):
    Zero out rows [h*64:(h+1)*64] of c_proj.weight (shape [768,768], row-major).
    c_proj.weight is a Conv1D, actual shape is [768, 768] transposed from Linear.
    Verify: `model.transformer.h[i].attn.c_proj.weight.shape` and transpose semantics.
  Compute perplexity on held-out probe.

### Step 7 — Save results
`autoresearch/results/cycle_022_head_census.json`: all per-head metrics + interventions
`autoresearch/results/cycle_022_head_census.png`: 3 subplots — eff_rank heatmap, variation
  heatmap, prunability heatmap (12×12 grid, rows=layers, cols=heads)

## Evaluation
- Count heads with eff_rank < 19.2 (= 0.3 × 64). Report fraction.
- Report ppl change at K=5,10,20,30.
- Classify: CONFIRMED / REFUTED / INCONCLUSIVE per criteria in brief.
