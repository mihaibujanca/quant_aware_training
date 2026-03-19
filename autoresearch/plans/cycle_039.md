# Cycle 039 Plan — Inter-block Residual-Stream lstsq Correction

## Goal
Fit a single affine correction C at the residual stream after the last compressed block, fold it into the next layer's weights, and measure PPL recovery for k=5 at 6x.

## Rationale
k=5 at 6x gives +16-17% PPL while k=4 gives +9-10%. The ~7pp jump suggests compounding error accumulates in the residual stream. If the error is mostly mean-shift + low-rank distortion, a single lstsq correction can recover several pp.

## Script
New file: `experiments/ffn_residual_correction_k5.py` (adapt from `ffn_block_ordering_k5.py`).

## Procedure

### Step 1 — Setup (reuse standard pipeline)
Load GPT-2, build probe corpus (200 train / 50 holdout / 50 eval), compute baseline PPL. Compress all 6 blocks at 6x. Use best k=5 combo from cycle 038: blocks {7,5,1,3,6} (drop 4).

### Step 2 — Collect residual-stream hidden states
Register a forward pre-hook on the transformer block immediately AFTER the last compressed block (by layer index). For each probe_train batch:
- Run float model -> collect `h_float[batch, seq, 768]`
- Run compressed model -> collect `h_compressed[batch, seq, 768]`
Flatten to `[N_tokens, 768]`. Use `probe_train` sequences (not eval_val).

The "last compressed block" = max(compressed_block_ids). The correction point = block `max_id + 1`. Hook target: `model.transformer.h[max_id + 1]` forward pre-hook (intercepts input to the next block = residual stream state).

### Step 3 — Fit affine correction via lstsq
Solve `h_float = h_compressed @ W_corr.T + b_corr` via torch.linalg.lstsq:
```python
# Augment h_compressed with ones column for bias
H = torch.cat([h_compressed, torch.ones(N, 1)], dim=1)  # [N, 769]
sol = torch.linalg.lstsq(H, h_float).solution              # [769, 768]
W_corr = sol[:768].T   # [768, 768]
b_corr = sol[768]      # [768]
```
Report: Frobenius norm of (W_corr - I), norm of b_corr, lstsq residual MSE.

### Step 4 — Fold correction into next layer's weights
GPT-2 block structure: `ln_1 -> attn -> + residual -> ln_2 -> mlp -> + residual`. The correction at block input is `h' = W_corr @ h + b_corr`. Fold into the LayerNorm + attention weights of block `max_id + 1`:
- Option A (clean): Insert a forward pre-hook that applies `h = h @ W_corr.T + b_corr` on the block input. Zero extra params at eval (it's a fixed transform).
- Option B (fold into LN): Modify `ln_1.weight` and `ln_1.bias` — but LayerNorm has normalization, so folding is non-trivial. Stick with Option A for correctness.

Use Option A (hook). This is zero learned params — the correction is computed once and frozen.

### Step 5 — Evaluate
Measure PPL of corrected model on `eval_val`. Compare:
| Config           | Expected PPL range |
|------------------|--------------------|
| Baseline (float) | ~120               |
| k=5 uncorrected  | ~140 (+16-17%)     |
| k=5 corrected    | target < 137 (< +14%) |
| k=4 reference    | ~132 (+9-10%)      |

Also run correction on k=4 (same procedure) to check for overcorrection/degradation.

### Step 6 — Diagnostic: correction rank analysis
SVD of (W_corr - I) to see effective rank. If rank << 768, the error is low-dimensional and future work could use a smaller correction.

## Verdict Criteria
- **CONFIRMED**: k=5 corrected PPL gap <= +14% (>= 3pp reduction from +17%).
- **REFUTED**: correction reduces gap by <1pp.
- **INCONCLUSIVE**: helps k=5 but worsens k=4 by >1pp.

## Outputs
- `autoresearch/results/cycle_039_residual_correction.json`
- `autoresearch/results/cycle_039_residual_correction.png` (bar chart: baseline / k4 / k5 / k5-corrected)

## Estimated cost
~8 min: load + probe + compress + 2x full forward for hidden collection + lstsq + 3 evals.
