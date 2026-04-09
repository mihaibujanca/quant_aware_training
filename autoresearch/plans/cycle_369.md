# Plan — Cycle 369 / Item 360

## Pre-flight

- **P1 (duplicate):** PASS. Cycle 007 (item 12) tested ordering on toy models only. Cycle 038 (item 57) tested block *selection* (which 5 of 6), not accumulation *ordering* — and explicitly noted "insertion order is a no-op" because that pipeline compressed all blocks from float activations independently. No GPT-2 ordering experiment with the current topk+lstsq+adapter recipe exists.
- **P2 (dependency):** PASS. Item 355 is marked "confirmed" in item 360's depends-on. Although item 355 is not found by that number in PAST (likely renumbered), its substance is covered: multi-block topk+lstsq works (cycle 349 item 358, cycle 354 item 362), block importance ranking exists (cycle 367 item 376).
- **P3 (one cycle):** PASS. 3 orderings × 4 stages × 3 seeds = 36 runs, but each run is one forward+lstsq+500 Adam steps. Total ~2000 steps × 9 seed-ordering combos. Template code exists.
- **P4 (program.md):** No changes needed.
- **P5 (null baseline):** The null hypothesis is path-independence (all orderings converge to same PPL at stage 4). Threshold >5% gap falsifies this.
- **P6 (method alignment):** Consistent with the working recipe (topk k=512, lstsq, adapter h=64, Adam).
- **P7 (statistical power):** 3 seeds. At 4-block stage, cycle 365 showed ~0.13 PPL seed variance for uniform 6-block. A 5% gap on ~66 PPL baseline is ~3.3 PPL — well above expected seed noise (~0.3 PPL). Power is adequate.

## Design

### Critical design choice: sequential cumulative compression

Ordering only matters if compression is sequential (each block compressed from the *current* model state, not all from float). If we compress all blocks from float activations in a single forward pass, ordering is irrelevant by construction (cycle 038 proved this).

Protocol for each (ordering, seed) pair:
1. Load fresh GPT-2. Measure float PPL.
2. For stage s in [1, 2, 3, 4]:
   a. Collect MLP activations for the **single next block** from the current model state (hooks on one block only).
   b. Run topk k=512 + lstsq to get compressed c_fc / c_proj weights.
   c. Install CompressedGPT2MLPWithAdapter (adapter h=64).
   d. Train adapter + proj_weight for this block AND all previously-installed adapters jointly, 500 steps, lr=1e-4 Adam.
   e. Freeze the newly compressed block's base weights (keep adapter trainable for future stages).
   f. Measure cumulative PPL → record as ppl_stage_s.
3. Record all 4 stage PPLs + final 4-block PPL.

### Three orderings

Block importance deltas from cycle 367:
- Block 1: +1.60 (easiest)
- Block 7: +2.35
- Block 4: +6.85
- Block 11: +9.12 (hardest)

| Ordering | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---|---|---|---|---|
| (a) Easiest-first | b1 | b7 | b4 | b11 |
| (b) Block-index | b1 | b4 | b7 | b11 |
| (c) Hardest-first | b11 | b4 | b7 | b1 |

### Config

- `k = 512` (1/6 of 3072)
- `adapter_hidden = 64`
- `steps_per_stage = 500` (2000 total per ordering)
- `lr = 1e-4`
- `seeds = [42, 137, 2024]`
- `N_CALIB = 200, N_EVAL = 100, SEQ_LEN = 128`
- `DEVICE = mps`

### Verdict criteria

Compare mean PPL at stage 4 (4-block) across orderings:
- **CONFIRMED** if max(mean_ppl) / min(mean_ppl) > 1.05 (>5% gap between best and worst)
- **REFUTED** if the gap is < 3% (orderings are essentially equivalent)
- **INCONCLUSIVE** if gap is 3-5%

## Template

Base on `experiments/gpt2_adaptive_k_cycle374.py` for the core compression functions (topk+lstsq, adapter module, eval_ppl, data loading). Modify the main loop to do sequential single-block compression instead of all-at-once.

Key modifications from template:
1. `compress_blocks()` → `compress_single_block()` — hooks and compresses one block at a time from current model state.
2. Outer loop over orderings × seeds (9 runs total).
3. Per-stage: compress 1 block → joint train all current adapters 500 steps → freeze → measure PPL.
4. Record `results[ordering_name][seed] = [ppl_stage1, ppl_stage2, ppl_stage3, ppl_stage4]`.

## Output

- Script: `experiments/gpt2_ordering_sensitivity_cycle369.py`
- Results: `autoresearch/results/cycle_369_ordering_sensitivity.json`
- Run command: `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/gpt2_ordering_sensitivity_cycle369.py`

## Estimated runtime

9 runs × (4 stages × 500 steps + 4 eval passes) ≈ 18,000 training steps + overhead. Each step ~0.1s on MPS → ~30 min total. Data loading + lstsq overhead adds ~5 min. Total: ~35 min.
