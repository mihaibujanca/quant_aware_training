# Cycle 374 — Item 382: Error budget test — k=768 on 6-block composition (R4)

## Pre-flight

### P1: Duplicate check — PASS

Not a duplicate of c365 (which tested non-uniform/adaptive k allocation across 6 blocks with total budget 3072 = 6x512). This tests uniform k=768 per block (50% more channels per block, no budget constraint across blocks). The hypothesis is different: c365 asked whether smarter allocation helps; this asks whether reducing per-block error (via more channels) breaks the composition barrier.

### P2: Dependency check — PASS

No dependencies listed. All baselines available from prior cycles.

### P3: Falsifiability — PASS

- Confirmed: 6-block k=768 mean PPL < 55.0
- Refuted: 6-block k=768 mean PPL > 60.0
- Inconclusive: 55-60 range
- 3-block k=768 sanity check: should beat k=512 baseline of 47.43

### P4: Cost estimate — PASS

Two configurations x 3 seeds = 6 runs. Each: load GPT-2, compress blocks, 2000-step joint training. ~5 min per run on MPS. Total ~30 min. Acceptable.

## Plan

### Script: `experiments/gpt2_error_budget_k768_cycle374.py`

Fork `experiments/gpt2_adaptive_k_cycle374.py` with these changes:

1. **Config changes**:
   - `N_STEPS = 2000` (was 1500)
   - `K_UNIFORM = 768` (new constant)
   - Remove all adaptive-k machinery (`compute_adaptive_k`, `FROB_NORMS`, `ADAPTIVE_K`, `SQRT_K`, `TOTAL_BUDGET`, `K_MIN`)
   - `THREE_BLOCKS = [2, 1, 10]` (subset)
   - `SIX_BLOCKS = [2, 1, 10, 4, 3, 11]` (full set)
   - `OUTPUT_PATH = "autoresearch/results/cycle_374_error_budget_k768.json"`
   - Update reference constants: `THREE_BLOCK_K512_REF = 47.43`, `SIX_BLOCK_K512_REF = 65.54`

2. **Two experiment modes** (run sequentially, no CLI flags needed — run both):
   - **3-block k=768**: `run_experiment(k_per_block={2:768, 1:768, 10:768}, block_ids=THREE_BLOCKS, seeds=SEEDS, ...)`
   - **6-block k=768**: `run_experiment(k_per_block={b:768 for b in SIX_BLOCKS}, block_ids=SIX_BLOCKS, seeds=SEEDS, ...)`

3. **Adapt `run_experiment`**: Accept `block_ids` parameter instead of hardcoded `ALL_BLOCKS`. Pass `block_ids` to `compress_blocks` and `install_compressed_blocks`.

4. **Verdict logic** (printed at end, based on 6-block mean PPL):
   - `mean_ppl < 55.0` → CONFIRMED (composition barrier broken with more channels)
   - `mean_ppl > 60.0` → REFUTED (barrier persists despite lower per-block error)
   - else → INCONCLUSIVE

5. **Output JSON structure**:
   ```json
   {
     "float_ppl": ...,
     "three_block_k768": { "mode": ..., "k_per_block": ..., "mean_ppl": ..., "std_ppl": ..., "seed_results": [...] },
     "six_block_k768": { ... },
     "verdict": "CONFIRMED|REFUTED|INCONCLUSIVE",
     "comparison": {
       "three_block_k512_ref": 47.43,
       "six_block_k512_ref": 65.54,
       "three_block_k768_improvement": ...,
       "six_block_k768_improvement": ...
     }
   }
   ```

### Execution

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/gpt2_error_budget_k768_cycle374.py
```

### Verification

- Float baseline must match 53.89 +/- 0.5
- 3-block k=768 PPL < 47.43 (must beat k=512; if not, something is wrong)
- Report per-block reconstruction error (frob_recon) alongside PPL for both configs
- All 3 seeds must run; report mean +/- std

### Analysis

Compare compression ratios and PPL:
- k=512: 3072→512, 6.0x compression. 3-block=47.43, 6-block=65.54. Delta=18.11
- k=768: 3072→768, 4.0x compression. 3-block=?, 6-block=?. Delta=?

Key question: does the 3-to-6 block PPL gap shrink proportionally with per-block error, or is there a nonlinear composition effect?
