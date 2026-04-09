# Cycle 376 — Item 384: Per-token CKA and trajectory divergence — 3-block vs 6-block

## Pre-flight

### P1: Duplicate check — PASS
c024 computed CKA across float layers only (layer-vs-layer similarity matrix). No prior cycle compares CKA between float and compressed models at matching layers.

### P2: Dependency check — CONDITIONAL PASS
Item says "depends on trained 3-block and 6-block models from c359/c357." No saved weights exist. Plan trains both as step 1, same as c371/c374/c375.

### P3: Falsifiability — PASS
Confirmed: CKA gap > 0.1 at any layer. Refuted: max gap < 0.05. Inconclusive: [0.05, 0.1].

### P4: Cost estimate — PASS
Two training runs (500 steps each) + 3 forward passes for hidden collection. ~15 min on MPS.

## Plan

### Script: `experiments/gpt2_cka_divergence_cycle376.py`

Fork from `experiments/gpt2_residual_geometry_cycle371.py`. Reuse: data loading, `compress_blocks`, `install_compressed_blocks`, `train_adapters`, `eval_ppl`, `CompressedGPT2MLPWithAdapter`.

1. **Config**: `BLOCKS_3 = [2,1,10]`, `BLOCKS_6 = [2,1,10,4,3,11]`, `K=512`, `ADAPTER_HIDDEN=64`, `N_STEPS=500`, `LR=3e-4`, `SEED=42`, `N_CKA=200` sequences for CKA measurement, `SEQ_LEN=128`. Output: `autoresearch/results/cycle_376_cka_divergence.json`.

2. **Training phase**: Train 3-block and 6-block models exactly as c371. Verify PPLs in expected ranges (3-block: [45, 52], 6-block: [60, 70]).

3. **Hidden collection** (new function `collect_all_hidden`):
   - For each of {float, 3-block, 6-block}: register forward hooks on all 12 `model.transformer.h[i]` blocks capturing output[0] (shape: batch, seq, 768).
   - Run N_CKA calibration sequences. Concatenate to (N_tokens, 768) per layer.
   - Subsample to 4096 tokens if needed (CKA is O(n*d^2), manageable).

4. **CKA computation** (reuse `linear_cka` from c024):
   ```python
   def linear_cka(X, Y):
       X, Y = X - X.mean(0), Y - Y.mean(0)
       hsic_xy = (X.T @ Y).pow(2).sum()
       hsic_xx = (X.T @ X).pow(2).sum()
       hsic_yy = (Y.T @ Y).pow(2).sum()
       return (hsic_xy / (hsic_xx * hsic_yy).sqrt()).item()
   ```
   Compute `cka_3block[l] = linear_cka(float_h[l], comp3_h[l])` and `cka_6block[l]` for l in 0..11.

5. **Per-token L2 divergence** (new):
   For each layer l, each config: `diffs = (float_h[l] - comp_h[l]).norm(dim=1)`. Report `mean`, `p95 = quantile(0.95)`, `max`.

6. **Plot** (`autoresearch/results/cycle_376_cka_divergence.png`): 2-panel figure.
   - Top: CKA vs layer (two lines: 3-block blue, 6-block red). Y-axis [0, 1].
   - Bottom: L2 mean + shaded p95 band vs layer.

7. **Output JSON**:
   ```json
   {"float_ppl": ..., "ppl_3block": ..., "ppl_6block": ...,
    "cka_3block": [12 floats], "cka_6block": [12 floats],
    "l2_3block": [{"mean":..,"p95":..,"max":..} x12],
    "l2_6block": [{"mean":..,"p95":..,"max":..} x12],
    "max_cka_gap": ..., "gap_layer": ..., "verdict": "..."}
   ```

8. **Verdict**: `max_gap = max(cka_3block[l] - cka_6block[l] for l)`.
   - gap > 0.1 → CONFIRMED. gap < 0.05 → REFUTED. else → INCONCLUSIVE.

### Execution

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/gpt2_cka_divergence_cycle376.py
```

### Verification
- Float PPL must match 53.89 ± 0.5.
- 3-block PPL in [45, 52], 6-block in [60, 70].
- CKA values in [0, 1]. CKA at layer 0 (embedding, uncompressed) should be ~1.0 for both.
- L2 at layer 0 should be ~0 for both (no compression applied yet).
