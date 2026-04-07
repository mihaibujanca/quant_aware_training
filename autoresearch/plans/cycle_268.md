# Cycle 268 — Item 278

## Goal
K=64 4-block sweep — extend cycle 257's K=64 5-block (62.181) to 4 blocks. Test SUBSETS=[[1,2,4,7],[1,4,7,8],[1,2,7,8]] at K=64.

## Hypothesis
4-block K=64 absorbs the smaller K well → best_mean ≤ 56.0 (between cycle 257 K=64 5-block 62.18 and cycle 255 K=512 4-block 49.61, weighted toward K=512 by the concave-up curve).

## Setup
Trivial K swap on cycle 265 template. New script `experiments/four_block_k64_sweep_cycle268.py`. K=64; SUBSETS=[[1,2,4,7],[1,4,7,8],[1,2,7,8]]; W_out+b_out trainable joint Adam LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore; 3 seeds {42,137,2024}; disjoint calib=all_ids[:200]/eval=all_ids[200:300]; SEQ_LEN=128; MPS; GPT-2 small; wikitext-2 val. Trainable per config ≈ 197,376 = 4 × (49,344 = 64×768+768).

## Verdict thresholds (item 278 verbatim)
CONFIRMED if min over subsets best_mean ≤ 56.0; REFUTED if all subsets best_mean ≥ 65.0; INCONCLUSIVE otherwise.

## Baselines
cycle 257 K=64 5-block 62.181; cycle 265 K=128 4-block min 53.647; cycle 255 K=512 4-block min 49.609; cycle 230 K=64 3-block 50.781; float 53.895.

## Run cap
~10 min wall (3 subsets × 3 seeds = 9 fits; cycle 265 wall was 21 min at K=128, K=64 should be smaller).

## Artifacts
`experiments/four_block_k64_sweep_cycle268.py`, `autoresearch/results/cycle_268_four_block_k64_sweep.json`, `autoresearch/results/cycle_268_summary.md`, `autoresearch/logs/cycle_268_run.log`, `autoresearch/results/cycle_268_manifest.txt`.

## Verification
JSON+log present; verdict matches threshold logic; assert restore matches recorded best (script-internal).
