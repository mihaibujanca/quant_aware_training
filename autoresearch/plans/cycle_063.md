# Cycle 063 Plan: Stacking-Aware Boundary Correction (Item 79)

## Goal
Fit per-block affine corrections C_i = A_i·h + b_i in cascade order on STACKED activations (not float). Cycle 043 fitted on float activations and corrections interfered. The fix: C_1 on float, C_2 on (compressed_1+C_1) output, etc.

## File
`experiments/stacking_aware_correction_gpt2.py` — new, based on `ffn_multipoint_correction.py` + `cascaded_compression_gpt2.py`.

## Config
```
TARGET_BLOCKS = [1, 4, 5, 7]
HIDDEN_FRACTION = 1/6          # 6x compression
SEEDS = [42, 123, 7]
PROBE_TRAIN_SEQS = 200         # → 200×128 = 25600 tokens for probe/correction
EVAL_VAL_SEQS = 200
SEQUENCE_LENGTH = 128
BATCH_SIZE = 4
CORRECTION_HELD_OUT_FRAC = 0.2 # 80/20 split for fit vs held-out MSE
OWT_PATH = data/openwebtext_sample.txt
```

## Data feeding for `fit_correction`
All probe tokens (PROBE_TRAIN_SEQS × SEQUENCE_LENGTH = 25 600) are collected per correction point. **No cap at 4096.** Before fitting, randomly split 80/20 into fit/held-out sets (per seed). Fit lstsq on the 80% split. Compute and log held-out MSE on the 20% split. With d=768, the affine has 769 parameters per output dim; 20 480 fit tokens gives ~26:1 overdetermination, well above the 5:1 floor.

## Experiment loop (per seed)
1. Load GPT-2, build OWT probe corpus, eval float baseline PPL.
2. Probe float geometry, compress all 4 blocks at 6x (topk+lstsq). Eval uncorrected stacked PPL (must replicate ~+30%).
3. **Naive corrections (cycle 043 replication)**:
   - `comp_model = swap_compressed_ffns(teacher, pack)`.
   - For block_id in [1,4,5,7]: collect `h_float` from teacher, `h_comp` from comp_model, 80/20 split, fit lstsq on 80%, log held-out MSE, install hook.
   - Eval PPL → `ppl_naive[seed]`.
4. **Stacking-aware corrections**:
   - `comp_model = swap_compressed_ffns(teacher, pack)` (fresh copy).
   - For block_id in [1,4,5,7]:
     a. Collect `h_float` from teacher at block_id+1.
     b. Collect `h_comp` from comp_model at block_id+1 (prior corrections already installed).
     c. 80/20 split. Fit `[h_comp, 1] @ [W; b] ≈ h_float` via lstsq on 80%.
     d. Log held-out MSE. Install correction hook on comp_model.
   - Eval PPL → `ppl_cascaded[seed]`.
5. Aggregate mean/std over 3 seeds.

## Diagnostics
Per correction point, log to JSON: `block_id`, `fit_mse`, `held_out_mse`, `n_fit`, `n_held_out`, `overdetermination_ratio`. Flag if held-out MSE > 1.5× fit MSE (overfitting signal).

## Sanity checks
- Uncorrected baseline must match cycle 058 (+30.2% ± 3pp).
- Naive corrections should roughly match cycle 043 behavior (interference at k=4).
- Block 1 correction identical between naive and cascaded (no prior blocks).

## Outputs
- `autoresearch/results/cycle_063_stacking_correction.json` — includes per-block diagnostics.
- `autoresearch/results/cycle_063_stacking_correction.png` — bar chart: float, uncorrected, naive-corrected, cascaded-corrected PPL with error bars.

## Verdict
- **CONFIRMED**: cascaded corrections ≤ +12% PPL degradation.
- **REFUTED**: cascaded corrections worse than uncorrected (+30.2%).
- **INCONCLUSIVE**: 12–30% range.
