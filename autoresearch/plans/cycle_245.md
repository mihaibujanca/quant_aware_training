# Cycle 245 — Item 255: Drop block 5 from 6-block compression

## Hypothesis

> 255 (P0): Drop block 5 from 6-block compression — can we beat the 59.5 floor by dropping cycle 243's biggest contributor? (empirical)
> Recipe: BLOCKS=[1,2,4,7,8] K=512, trainable W_out+b_out, topk+lstsq init, joint Adam LR=1e-4, 4000 steps, EVAL_EVERY=100, best-state restore, disjoint calib=all_ids[:200] / eval=all_ids[200:300], SEQ_LEN=128, 3 seeds. Compare to cycle 221 6-block 59.515 ± 0.094 and cycle 242 3-block early 47.135 and cycle 210 3-block [1,4,7] 46.556.
> Confirmed: best_mean ≤ 54.0. Refuted: best_mean ≥ 57.0. Inconclusive: (54.0, 57.0).

## Approach

Copy `experiments/adam_6block_earlystop_cycle221.py` to `experiments/drop_block5_cycle245.py`. Only change: `BLOCKS = [1, 2, 4, 7, 8]` (5 blocks). All other hyperparameters identical. Update OUTPUT_PATH, SUMMARY_PATH, cycle/item stamps, verdict thresholds.

## Files

- NEW `experiments/drop_block5_cycle245.py`
- NEW `autoresearch/results/cycle_245_drop_block5.json`
- NEW `autoresearch/results/cycle_245_summary.md`

## Constraints

- BLOCKS = [1, 2, 4, 7, 8]
- K=512, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100
- SEEDS = [42, 137, 2024]
- Disjoint calib=all_ids[:200] / eval=all_ids[200:300]
- topk+lstsq init → joint Adam on W_out+b_out; best-state snapshot+restore; assert restore within 1e-3
- Anchors: cycle 221 6-block [1,2,4,5,7,8] = 59.515 ± 0.094; cycle 210 3-block [1,4,7] = 46.556
- Cycle/item stamped in JSON: cycle=245, item=255
- Do NOT overwrite cycle 221 artifacts

## Verdict logic

- CONFIRMED if best_mean ≤ 54.0 (beats cycle 221 by ≥5.5 PPL — block 5 was the dominant contributor)
- REFUTED if best_mean ≥ 57.0 (floor only moves ≤2.5 PPL — cascade-joint, no single-block remediation)
- INCONCLUSIVE if 54.0 < best_mean < 57.0

## Steps

1. Copy `experiments/adam_6block_earlystop_cycle221.py` to `experiments/drop_block5_cycle245.py`.
2. Change `BLOCKS` literal to `[1, 2, 4, 7, 8]`.
3. Update OUTPUT_PATH to `autoresearch/results/cycle_245_drop_block5.json`, SUMMARY_PATH to `autoresearch/results/cycle_245_summary.md`.
4. Stamp cycle=245, item=255 in JSON payload.
5. Rewrite verdict block with thresholds 54.0 / 57.0 and ΔPPL vs cycle 221 (59.515) and vs cycle 210 (46.556).
6. Summary markdown: per-seed best PPL + best step, init PPL, trainable param count, mean ± std, ΔPPL vs 59.515 and vs 46.556, verdict.
7. Do not execute — runner will schedule.

## Risks

- 5-block joint fit on MPS: memory footprint slightly below cycle 221 (5 vs 6 blocks); should fit.
- Cycle 244 (just-completed) showed mid-blocks are not dominant in fresh fits, so the cycle 243 per-block attribution may not predict this result; verdict may land INCONCLUSIVE.
- One-shot 4-block [1,2,4,8] and [1,2,7,8] controls deferred — decide after main result clears the gate (not in this plan).

## Verification

- Script parses cleanly: `python -c "import ast; ast.parse(open('experiments/drop_block5_cycle245.py').read())"`.
- `BLOCKS` literal equals `[1, 2, 4, 7, 8]`; `K == 512`; `LR == 1e-4`; `N_STEPS == 4000`; `EVAL_EVERY == 100`; `SEEDS == [42, 137, 2024]`.
- OUTPUT_PATH ends with `cycle_245_drop_block5.json`; no collision with cycle 221 artifacts.
- Cycle/item fields present in JSON payload.
