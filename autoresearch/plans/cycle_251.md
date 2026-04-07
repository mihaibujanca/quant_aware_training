# Cycle 251 — Item 261: 5-block [1,2,4,7,8] at K=128

## Hypothesis

> 261 (P0): 5-block [1,2,4,7,8] at K=128 — does cascade tolerance survive lower channel count?

## Approach

Copy `experiments/drop_block5_cycle245.py` to `experiments/five_block_k128_cycle251.py`. Only change: `K = 128` (was 512). BLOCKS unchanged. Update paths and cycle/item stamps.

## Files

- NEW `experiments/five_block_k128_cycle251.py`
- NEW `autoresearch/results/cycle_251_five_block_k128.json`
- NEW `autoresearch/results/cycle_251_summary.md`

## Constraints

- BLOCKS = [1, 2, 4, 7, 8]; K = 128
- N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024]
- Disjoint calib=all_ids[:200] / eval=all_ids[200:300]
- topk+lstsq init → joint Adam W_out+b_out; best-state restore; assert within 1e-3
- Anchors: cycle 245 5-block K=512 = 53.826; cycle 231 6-block K=128 = 67.562; cycle 221 6-block K=512 = 59.515
- Cycle/item stamped: cycle=251, item=261
- Do NOT overwrite cycle 245 artifacts

## Verdict logic

- CONFIRMED if best_mean ≤ 56.0 (K=128 penalty stays small at 5 blocks)
- REFUTED if best_mean ≥ 62.0 (K=128 penalty scales like 6-block)
- INCONCLUSIVE if 56.0 < best_mean < 62.0

## Steps

1. Copy cycle 245 script to `experiments/five_block_k128_cycle251.py`.
2. Change `K` literal to `128`.
3. Update OUTPUT_PATH / SUMMARY_PATH to cycle_251_* paths.
4. Stamp cycle=251, item=261 in JSON payload.
5. Rewrite verdict block (56.0 / 62.0) and ΔPPL vs 53.826 (cycle 245) and 67.562 (cycle 231).
6. Summary: per-seed best PPL+step, init PPL, param count, mean±std, ΔPPL vs anchors, verdict.
7. Do not execute — runner will schedule.

## Risks

- 5-block K=128 joint fit: memory well below cycle 221; fits trivially.
- If K=128 penalty is block-count-linear, expect ~61 (53.8 + 7*5/6). Could land INCONCLUSIVE.

## Verification

- `python -c "import ast; ast.parse(open('experiments/five_block_k128_cycle251.py').read())"`.
- `BLOCKS == [1,2,4,7,8]`; `K == 128`; LR/N_STEPS/EVAL_EVERY/SEEDS unchanged.
- OUTPUT_PATH ends `cycle_251_five_block_k128.json`; no collision with cycle 245.
- Cycle/item fields present in JSON payload.
