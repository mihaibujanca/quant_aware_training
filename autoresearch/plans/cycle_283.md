# Cycle 283 Plan — Item 293

## Item (verbatim)
293. **1-block [2] K sweep — K ∈ {16, 32, 64} — can rank-1 block 2 compress to very small K?**
- Mirror cycle 276 (`experiments/one_block_k_sweep_cycle276.py`) verbatim. BLOCKS=[2], K_VALUES=[16, 32, 64]. 3 seeds {42, 137, 2024}, 9 fits.
- CONFIRMED iff K=16 best_mean ≤ 52.0 OR K=32 best_mean ≤ 50.0. REFUTED iff all K ≥ 55.0. Else INCONCLUSIVE.

## Approach
Trivial swap of cycle 276 template: BLOCKS=[2], K_VALUES=[16, 32, 64], OUTPUT_PATH/SUMMARY_PATH retargeted to cycle 283. Verdict logic rewritten per item 293. Same Adam W_out+b_out, lstsq init, 4000 steps, eval-100, best-state restore.

## Files
- `experiments/block2_k_sweep_cycle283.py` (new)
- `autoresearch/results/cycle_283_block2_k_sweep.json`, `_summary.md`, `_manifest.txt`
- `autoresearch/logs/cycle_283_run.log`

## Config
3 seeds × 3 K = 9 fits. ~2.5 min/fit ≈ 23 min wall. Disjoint N_CALIB=200/N_EVAL=100, SEQ_LEN=128, MPS, GPT-2 small, wikitext-2 val.

## Verification
Assert trainable params = K×768+768 per K. Restore == best within 1e-3. Init PPL deterministic across seeds. Verdict per item 293.

## Baselines to record
Float 53.895; cycle 276 [1] K=32/64/128/256 best_means; cycle 281 [1,2] K=512=44.339; cycle 271 [1] K=512=43.821.
