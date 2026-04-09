# Cycle 381 — Item 389: 6-block compressed+adapter composition penalty

## Pre-flight

### P1: Duplicate check — PASS
No prior cycle measures 6-block float+adapter or 6-block compressed+adapter on test data.

### P2: Dependency check — PASS
Depends on c378 (CONFIRMED, established test protocol and 3-block reference numbers).

### P3: Falsifiability — PASS
Confirmed: 6-block penalty < 15 PPL. Refuted: > 25 PPL. Inconclusive: [15, 25].

### P4: Cost estimate — PASS
2 configs × 3 seeds × 2000 steps × ~0.15s/step ≈ 30 min on MPS. 6 blocks adds ~2× forward vs 3-block, so ~60 min total.

## Plan

### Script: `experiments/gpt2_6block_penalty_cycle381.py`

Fork from `experiments/gpt2_heldout_test_cycle378.py`. Key changes:

1. **BLOCKS**: Change `[2, 1, 10]` to `[2, 1, 10, 4, 3, 11]`.

2. **Adapter config**: Keep FLOAT_ADAPTER_H=64, COMPRESSED_ADAPTER_H=128 (same as c378 for direct comparison).

3. **Expected params**: Update counts for 6 blocks.
   - Float adapter: 6 × (2×64×768 + 64 + 768) = 6 × 99,136 = 594,816.
   - Compressed adapter: 6 × (2×128×512 + 128 + 768) = 6 × 131,968 = 791,808 (+ proj weights).

4. **Thresholds**: CONFIRMED_LT_PENALTY=15.0, REFUTED_GT_PENALTY=25.0. Penalty = mean_comp_test - mean_float_test.

5. **Float baseline on test**: Eval unmodified GPT-2 on test. Expect ~56.52 (sanity check vs c378).

6. **Per-seed loop** (42, 137, 2024):
   - **Config A (float+adapter)**: Load fresh GPT-2, attach `FloatGPT2MLPWithAdapter` (h=64) to all 6 blocks. Freeze all, unfreeze adapter params. Train 2000 steps on val, select best by val eval, eval once on test.
   - **Config B (compressed+adapter)**: Load fresh GPT-2, collect calibration activations, build topk+lstsq compression (k=512) for all 6 blocks. Attach `CompressedGPT2MLPWithAdapter` (h=128). Freeze all, unfreeze proj+adapter. Train 2000 steps, select best by val eval, eval once on test.

7. **Output JSON**: `autoresearch/results/cycle_381_6block_penalty.json`
   ```json
   {"float_test_ppl": ...,
    "float_adapter_6block": {"mean_test_ppl": ..., "std": ..., "seeds": [...]},
    "compressed_adapter_6block": {"mean_test_ppl": ..., "std": ..., "seeds": [...]},
    "penalty_6block": ...,
    "penalty_3block_ref": 7.49,
    "verdict": "...", "confirmed_lt": 15.0, "refuted_gt": 25.0}
   ```

8. **Verdict**: penalty < 15 → CONFIRMED (sublinear). > 25 → REFUTED (superlinear). Else INCONCLUSIVE.

### Execution

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/gpt2_6block_penalty_cycle381.py
```

### Verification
- Float test PPL matches c378 baseline (~56.52 +/- 0.5).
- Init PPL (step 0) for float+adapter matches float test PPL +/- 0.1.
- Trainable param counts match expectations (594,816 for float adapter, ~791,808+proj for compressed).
- Both configs converge (PPL decreasing over 2000 steps).
- Compare 6-block penalty to 3-block penalty (7.49) to assess scaling.
