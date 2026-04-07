# Cycle 247 — Item 257: Heterogeneous K across blocks

## Item (verbatim)
> 257 (P1): Heterogeneous K — K=512 on binding blocks 4,5; K=128 on 1,2,7,8.
> Test whether the K=128 penalty (cycle 231: 67.6) concentrates on cycle 243's mid-block contributors.

## Hypothesis
Cycle 243 showed blocks 4,5 as dominant cost contributors in the joint cascade. Giving them K=512 while keeping cheap K=128 on the easy blocks (1,2,7,8) should close most of the K=128 → K=512 gap (67.6 → 59.5) at ~1.18M params instead of 2.36M.

## Source files
- Base: `experiments/adam_6block_earlystop_cycle221.py` (K=512 6-block, 59.515).
- Mirror: `experiments/adam_6block_k128_cycle231.py` (K=128 6-block, 67.562).
- New script: `experiments/hetero_k_6block_cycle247.py`.

## Implementation steps
1. Copy cycle 231 script to `experiments/hetero_k_6block_cycle247.py`.
2. Replace scalar `K = 128` with `K_PER_BLOCK = {1: 128, 2: 128, 4: 512, 5: 512, 7: 128, 8: 128}`. Keep `BLOCKS = [1, 2, 4, 5, 7, 8]`.
3. In `run_seed`, the per-block `fit_block_lstsq` call already takes `K` as an argument — pass `K_PER_BLOCK[b]` instead of the scalar. `CompressedMLP` is per-block so heterogeneous hidden widths work unchanged.
4. Joint Adam over `W_out + b_out` for all 6 blocks (trainable params auto-size from the heterogeneous `W_out` shapes). Unchanged loop, snapshot, restore, assertion.
5. Update `OUTPUT_PATH`, `SUMMARY_PATH`, cycle/item numbers (247/257), baselines (221 K=512 → 59.515, 231 K=128 → 67.562), verdict thresholds.
6. Persist `K_PER_BLOCK` into the result JSON `config` dict and summary header.

## Config (unchanged from 221/231)
N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024]. Disjoint calib `all_ids[:200]`/eval `[200:300]`. Best-state snapshot + restore with assertion.

## Expected trainable count
4×(128×768 + 768) + 2×(512×768 + 768) = 393,728 + 787,968 = **1,181,696** (~1.18M). Assert in script.

## Verdict
- CONFIRMED if `best_mean ≤ 61.0`
- REFUTED if `best_mean ≥ 65.0`
- INCONCLUSIVE if 61.0 < best_mean < 65.0

## Outputs
- `autoresearch/results/cycle_247_hetero_k.json`
- `autoresearch/results/cycle_247_summary.md`
DO NOT execute — plan only; runner picks up the script.
