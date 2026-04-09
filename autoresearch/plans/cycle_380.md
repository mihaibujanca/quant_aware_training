# Cycle 380 — Item 388: Full-model float+adapter ceiling (all 12 blocks)

## Pre-flight

### P1: Duplicate check — PASS
No prior cycle trains adapters on all 12 GPT-2 blocks. c377 used 3 blocks only.

### P2: Dependency check — PASS
Depends on c377 (CONFIRMED) and c378 (CONFIRMED, established test protocol). Template exists.

### P3: Falsifiability — PASS
Confirmed: 12-block PPL < 36.0. Refuted: PPL > 38.0. Inconclusive: [36, 38].

### P4: Cost estimate — PASS
3 seeds × 2000 steps × ~0.15s/step ≈ 15 min on MPS. 12 blocks adds forward cost (~2× vs 3-block), so ~30 min total.

## Plan

### Script: `experiments/gpt2_adapter_12block_cycle380.py`

Fork from `experiments/gpt2_adapter_only_cycle377.py`. Key changes:

1. **BLOCKS**: Change `[2, 1, 10]` to `list(range(12))` (all 12 blocks).

2. **Data splits (c378 protocol)**: Train on wikitext-2-val (N_CALIB=200), checkpoint selection on val eval (50 seqs from val), final eval on wikitext-2-test (N_EVAL=200, touched once per seed after checkpoint selection).

3. **Expected params**: 12 × (2×64×768 + 64 + 768) = 12 × 99,136 = 1,189,632. Update EXPECTED_ADAPTER_PARAMS and assertion.

4. **Thresholds**: CONFIRMED_LT = 36.0, REFUTED_GT = 38.0.

5. **Float baseline on test**: Eval unmodified GPT-2 on test split. Expect ~56.52 (matching c378).

6. **Per-seed loop** (42, 137, 2024):
   - Load fresh GPT-2, attach `FloatGPT2MLPWithAdapter` to all 12 blocks.
   - Freeze all params, unfreeze adapter params only.
   - Train 2000 steps on val, select best checkpoint by val eval loss.
   - Evaluate best checkpoint once on test. Record test PPL.

7. **Output JSON**: `autoresearch/results/cycle_380_adapter_12block.json`
   ```json
   {"float_test_ppl": ...,
    "three_block_ref": {"val": 38.94, "test": 39.73},
    "twelve_block": {"mean_test_ppl": ..., "std": ..., "seeds": [...]},
    "verdict": "...", "confirmed_lt": 36.0, "refuted_gt": 38.0}
   ```

8. **Verdict**: mean_test_ppl < 36.0 → CONFIRMED. > 38.0 → REFUTED. Else INCONCLUSIVE.

### Execution

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/gpt2_adapter_12block_cycle380.py
```

### Verification
- Float test PPL should match c378 baseline (~56.52 ± 0.5).
- Init PPL (step 0) for 12-block adapter should match float test PPL ± 0.1 (adapters initialized near zero).
- Trainable param count must be exactly 1,189,632.
- Training should converge (PPL decreasing over 2000 steps, not diverging).
- Compare final PPL to 3-block test reference (39.73) to quantify headroom.
