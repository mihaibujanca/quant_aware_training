# Cycle 151 Plan: Correction Subspace Analysis on Unpolished Model (Item 161)

## Task
Redo cycle 142's correction subspace decomposition with the correct protocol: train LoRA corrections on the UNPOLISHED topk+lstsq model (no 500-step Adam polish). Cycle 142 trained corrections on a polished model (PPL ~71) and PPL worsened to 88.7 because corrections and polish compete. The unpolished model (PPL ~85) is where corrections have room to work, as demonstrated in cycle 115 (PPL 85 -> 59).

## Hypothesis
The correction learned on the unpolished model will split into a local-error-cancellation component (projection onto E_local = W_compressed - W_float error subspace) and a cascade-compensation component (orthogonal complement). If cascade compensation accounts for >=30% of correction energy and its removal degrades PPL by >=5%, corrections are partially addressing inter-block error propagation — relevant to the composition barrier.

## Pre-flight
- P1 (duplicate): cycle 142 attempted this but with wrong protocol (polished model). This fixes the method failure. Distinct.
- P2 (deps): none. Uses existing compression + correction infrastructure.
- P3 (scope): 1 seed, ~45 min. Fork cycle 142 script, remove polish.
- P4 (null baseline): float PPL ~57.65. Unpolished PPL should be ~85. Corrected PPL should land in [60, 75]. Abort if float PPL > 60 or unpolished PPL < 70 (would indicate polish is somehow active).
- P5 (param accounting): LoRA r=8: 4 blocks x 2 x 768 x 8 = 49,152 params. Assert at init.

## Method

### Step 1: Fork `experiments/correction_subspace_cycle142.py` to `experiments/correction_subspace_unpolished_cycle151.py`

Changes from cycle 142:
1. In `build_compressed_model()`: DELETE the polish loop (lines 108-131 of cycle 142). After `swap_compressed_ffns()`, return the model directly.
2. Update cycle/item numbers: cycle=151, item=161.
3. Update output path: `cycle_151_correction_subspace_unpolished.json`.
4. Add assertion: unpolished PPL must be in [75, 100]. If outside, abort.

Everything else stays identical: LowRankCorrection(768, 8), 2000 correction steps, lr=1e-4, blocks [7,5,4,1], seed 42.

### Step 2: Build unpolished compressed model
- Load GPT-2, eval float PPL. Abort if > 60.
- Compress blocks [7,5,4,1] at 6x via topk + lstsq. NO POLISH.
- Eval unpolished PPL. Should be ~85. Abort if outside [75, 100].

### Step 3: Train LoRA corrections (2000 steps)
- Identical to cycle 142 `train_corrections()`: freeze model, train only {U_l, V_l} per block.
- Adam lr=1e-4, batch_size=4, seq_len=128, CE loss on OWT.
- Eval corrected PPL. Should be in [60, 75].

### Step 4: Collect error subspaces
- Hook into each compressed block, collect residual stream outputs for float vs compressed on 200 holdout sequences.
- Per-block error matrix E_b = float_acts - comp_acts. Top-8 PCs = error subspace basis.
- This is local error (what topk+lstsq got wrong at each block individually).

### Step 5: Decompose corrections
Per block b:
1. Extract C_b = U_b @ V_b (768x768, rank 8).
2. C_local = proj_{error_basis}(C_b) — projection onto local error PCs.
3. C_cascade = C_b - C_local — orthogonal complement (cascade + novel).
4. Record Frobenius norm fractions: ||C_local||_F / ||C_b||_F, ||C_cascade||_F / ||C_b||_F.
5. Sanity: ||C_local||_F^2 + ||C_cascade||_F^2 = ||C_b||_F^2 (Pythagorean, check residual < 1e-6).

### Step 6: Ablation
Per block b, 3 conditions (ablate ONE block at a time, other blocks keep full correction):
1. **full**: original correction (control)
2. **local_only**: replace C_b with rank-8 SVD of C_local
3. **cascade_only**: replace C_b with rank-8 SVD of C_cascade

Measure PPL for each. Key metric:
- local_recovery = (uncorrected_ppl - local_only_ppl) / (uncorrected_ppl - full_ppl)
- cascade_recovery = (uncorrected_ppl - cascade_only_ppl) / (uncorrected_ppl - full_ppl)

### Step 7: Verdict
From item 161 agenda:
- **Confirmed** if cascade component accounts for >=30% of correction energy AND cascade-only ablation PPL is >=5 PPL points better than uncorrected (i.e., cascade alone does meaningful work).
- **Refuted** if corrections are >90% local error cancellation (cascade is negligible in both energy and ablation).
- **Inconclusive** if decomposition is unstable or results don't clearly fit either criterion.

## Output
- `experiments/correction_subspace_unpolished_cycle151.py`
- `autoresearch/results/cycle_151_correction_subspace_unpolished.json`

## Validation
- Float PPL: 57.65 +/- 1. Abort if > 60.
- Unpolished PPL: must be in [75, 100]. Key check that polish is NOT applied.
- Corrected PPL: must be in [60, 75]. If > 80, correction training failed — extend to 3000 steps.
- Pythagorean check: ||C_local||^2 + ||C_cascade||^2 = ||C||^2 within 1e-6.
- Ablation sanity: full correction PPL should be best. If local_only or cascade_only beats full, something is wrong.

## Fallbacks
- If correction training diverges or corrected PPL > 80: reduce lr to 5e-5, extend to 3000 steps.
- If unpolished PPL is outside [75, 100]: verify polish removal. If it's ~70, polish was not fully removed.

## Wall time estimate
- Compression (no polish): ~3 min.
- Correction training: ~15 min.
- Error collection + decomposition: ~5 min.
- Ablation: 4 blocks x 3 conditions x ~30 sec = ~6 min.
- Total: ~30 min.
