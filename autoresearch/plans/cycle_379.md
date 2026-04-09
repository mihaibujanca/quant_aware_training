# Cycle 379 — Item 387: Compression-accuracy Pareto curve (k x adapter h sweep)

## Pre-flight

### P1: Duplicate check — PASS
No prior cycle sweeps k and adapter h jointly. c377/c378 tested single (k, h) points.

### P2: Dependency check — PASS
Depends on c378 (CONFIRMED). Template script and compression infrastructure exist.

### P3: Falsifiability — PASS
Confirmed: compressed config within 2 PPL of float+adapter at <= 80% total FFN params. Refuted: all dominated.

### P4: Cost estimate — PASS
8 configs × 1 seed × 2000 steps × ~0.15s/step ≈ 30 min on MPS. h=0 configs skip training (~instant).

## Plan

### Script: `experiments/gpt2_pareto_sweep_cycle379.py`

Fork from `experiments/gpt2_heldout_test_cycle378.py`. Key changes:

1. **Sweep grid (8 configs)**:
   - Float (k=3072, no compression): `H_VALUES = [0, 32, 64, 128]` → 4 configs
   - Compressed (k=512): `H_VALUES = [0, 32, 64, 128]` → 4 configs
   - Total: 8 configs. k=512 chosen as the working recipe from c377/c378.

2. **Param counting (FFN-only)**: For each (k, h) config, compute total FFN params across 3 blocks:
   - Compressed block: `fc_w(768*k) + fc_b(k) + proj_w(k*768) + proj_b(768)` = `2*768*k + k + 768`. Adapter: `k*h + h + 768*h + 768` (input from GELU hidden dim=k). Total per block = sum.
   - Float block: `768*3072 + 3072 + 3072*768 + 768` = 4,721,664. Adapter: `768*h + h + 768*h + 768` (input from FFN output dim=768). Total per block = sum.
   - Multiply by 3 blocks. Store as `total_ffn_params` in results.
   - **Note**: Pareto comparison is on FFN params only. Both configurations share the same frozen 124M GPT-2 backbone, so FFN compression savings are ~2.5% of total model params. Any Pareto claim must be scoped to "at matched FFN params."

3. **Training protocol** (matching c378):
   - Train on **val** split, eval on **test** split.
   - Seed **42** only (single seed). Note: c378 showed std ~0.05–0.08 PPL across 3 seeds, small relative to the 2 PPL confirmation threshold. Pareto-optimal configs should be validated with 3 seeds in a follow-up.
   - h=0: no adapter, no training. Just eval compressed/float FFN on test.
   - h>0: attach adapter, train 2000 steps on val, select best checkpoint by val loss, eval on test.
   - For float h=0: eval unmodified GPT-2 on test (sanity-check against c378 float_test_ppl=56.52).
   - Record: `{"k": k, "h": h, "is_float": bool, "total_ffn_params": int, "test_ppl": float, "best_val_ppl": float}`.

4. **Pareto frontier**: After all configs, compute Pareto frontier (non-dominated set in FFN-params vs PPL space, lower is better on both axes). Mark each config as pareto-optimal or dominated.

5. **Plot**: Scatter plot of total_ffn_params (x) vs test_ppl (y). Color by compressed vs float. Draw Pareto frontier line. Save to `autoresearch/results/cycle_379_pareto.png`.

6. **Output JSON**: `autoresearch/results/cycle_379_pareto_sweep.json`
   ```json
   {"configs": [...], "pareto_frontier": [...],
    "verdict": "CONFIRMED|REFUTED|INCONCLUSIVE",
    "best_compressed": {"k": ..., "h": ..., "ppl": ..., "params": ...},
    "best_float": {"k": 3072, "h": ..., "ppl": ..., "params": ...}}
   ```

7. **Verdict**: Check if any compressed config is within 2 PPL of the nearest float config at <= 80% FFN params. CONFIRMED if yes, REFUTED if all compressed are Pareto-dominated by float.

### Execution

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/gpt2_pareto_sweep_cycle379.py
```

### Verification
- Float h=0 test PPL should match c378 float_test_ppl (~56.52).
- Float h=64 test PPL should match c378 float+adapter (~39.67 for seed 42).
- Compressed k=512 h=128 should match c378 compressed+adapter (~47.24 for seed 42).
- Param counts: spot-check k=512 h=64 block = 2*768*512+512+768 + 512*64+64+768*64+768 = 786,432+83,776 = 870,208 per block, 2,610,624 total.
- All h=0 configs must have PPL > any h>0 config at same k (adapters should help).
