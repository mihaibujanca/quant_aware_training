# Cycle 274 Plan — Item 284: K=64 5-block extended budget (8000 steps) — does cycle 257 plateau?

## Item (verbatim)
284 (P2): K=64 5-block extended budget (8000 steps) — does cycle 257 plateau? BLOCKS=[1,2,4,7,8] K=64, 8000 steps, 3 seeds.

## Hypothesis
K=64 5-block plateaus like K=128 5-block (cycle 254 noop) → best_mean ∈ [61.5, 62.5] (within ±0.5 PPL of cycle 257's 4k 62.181, no meaningful descent in second 4k).

## Verdict thresholds
- CONFIRMED (plateau) iff best_mean ∈ [61.5, 62.5].
- REFUTED (still descending) iff best_mean ≤ 60.7 (≥1.5 PPL drop vs cycle 257).
- INCONCLUSIVE otherwise (gap (60.7, 61.5) or > 62.5).

## Setup
Clone `experiments/five_block_k64_cycle257.py` → `experiments/five_block_k64_8k_cycle274.py`. BLOCKS=[1,2,4,7,8], K=64, joint Adam W_out+b_out, LR=1e-4, **N_STEPS=8000**, **EVAL_EVERY=200**, SEEDS=[42,137,2024], best-state restore, disjoint N_CALIB=200/N_EVAL=100, SEQ_LEN=128. Trainable=246,720 (5×(64×768+768)). 3 fits, ~6 min/fit ≈ 18 min total.

## Steps
1. Copy cycle 257 script to `experiments/five_block_k64_8k_cycle274.py`. Edit: `N_STEPS=8000`, `EVAL_EVERY=200`, `OUTPUT_PATH=autoresearch/results/cycle_274_five_block_k64_8k.json`, `SUMMARY_PATH=autoresearch/results/cycle_274_summary.md`. Add `CYCLE_257_K64_BASELINE=62.181`, `CYCLE_254_K128_8K_BASELINE=59.018`. Rename `final_ppl_at_step_4000` → `final_ppl_at_step_8000` (and printout). Update verdict logic to item 284 thresholds [61.5,62.5]/≤60.7.
2. Run `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/five_block_k64_8k_cycle274.py 2>&1 | tee autoresearch/logs/cycle_274_run.log`.
3. Per-seed assert restore<1e-3.
4. Save JSON + summary; write manifest with all per-seed/aggregate + cycle 257 + cycle 254 baselines.

## Baselines
cycle 257 K=64 5-block 4k = 62.181; cycle 254 K=128 5-block 8k = 59.018 (Δ vs cycle 251 4k = −0.101 noop); cycle 251 K=128 5-block 4k = 59.119; cycle 245 K=512 5-block = 53.826; cycle 269 K=1 6-block 16k = 91.61 (slow descent); float = 53.895.

## Bookkeeping
Delete item 284 from NEXT/P2; add `### Cycle 274` to PAST EXPERIMENTS top with verdict. Append TSV row. Stage cycle files only. Primary commit + bookkeeping commit + manifest. Audit.
