# Cycle 244 — Item 254: Sliding 3-block sweep

## Hypothesis

> 254 (P0): Sliding 3-block sweep — where does the cascade penalty bite?
> Test [3,4,5], [5,6,7], [6,7,8] at K=512 joint Adam W_out+b_out, 3 seeds. Map ΔPPL vs window start index.

## Approach

Copy `experiments/early_vs_late_3block_cycle242.py` to `experiments/sliding_3block_cycle244.py`. Swap `CONFIGS = [[3,4,5], [5,6,7], [6,7,8]]`. Keep recipe identical: topk+lstsq init → joint Adam on W_out+b_out, 3 seeds {42,137,2024}, best-state snapshot+restore. Update OUTPUT_PATH/SUMMARY_PATH and verdict block to iterate over all three windows.

## Files

- NEW `experiments/sliding_3block_cycle244.py`
- NEW `autoresearch/results/cycle_244_sliding_3block.json`
- NEW `autoresearch/results/cycle_244_summary.md`

## Constraints

- CONFIGS = [[3,4,5], [5,6,7], [6,7,8]]
- K=512, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100
- SEEDS=[42,137,2024]
- Anchors for comparison: cycle 210 [1,4,7]=46.556, cycle 242 [1,2,3]=47.135, cycle 242 [8,9,10]=55.192
- Best-state snapshot+restore; assert restore within 1e-3
- Cycle/item stamped in JSON: cycle=244, item=254

## Verdict logic

- CONFIRMED if any window mean_best ≤ 48.5
- REFUTED if all window mean_best ≥ 52.0
- INCONCLUSIVE otherwise

## Steps

1. Copy cycle 242 script to `experiments/sliding_3block_cycle244.py`.
2. Set CONFIGS to [[3,4,5], [5,6,7], [6,7,8]]; update OUTPUT_PATH, SUMMARY_PATH, cycle/item stamps.
3. Rewrite verdict block to iterate 3 windows: compute means, apply thresholds above, report Δ vs 46.556 and vs cycle 242 anchors [1,2,3]=47.135 and [8,9,10]=55.192.
4. Update summary markdown table headers/rows to cover all 3 windows plus anchor comparison row.
5. Do not execute — runner will schedule.

## Risks

- MPS memory pressure is unchanged (3 blocks at a time); same footprint as cycle 242.
- Window [5,6,7] overlaps with cycle 243 attribution showing block 5 is the heaviest contributor — may cluster near the bad end.

## Verification

- Script imports cleanly (`python -c "import ast; ast.parse(open('experiments/sliding_3block_cycle244.py').read())"`).
- CONFIGS literal equals [[3,4,5],[5,6,7],[6,7,8]].
- OUTPUT_PATH ends with `cycle_244_sliding_3block.json`.
- No overwrite of prior cycle_242/243 artifacts.
