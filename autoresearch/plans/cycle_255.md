# Cycle 255 — Item 265: 4-block K=512 sweep — [1,2,4,7], [1,4,7,8], [1,2,7,8]
## Hypothesis
> count-driven story holds → at least one 4-block subset achieves best_mean ≤ 51.0 (≥3 PPL below float, the count-prediction extrapolation from 5-block 53.826 minus another ~3 PPL).
## Approach
Adapt cycle 245 template into a 3-subset sweep. Per agenda item 265 spec: 3 seeds {42, 137, 2024} per subset, single script with config loop = 9 fits total (3 subsets × 3 seeds). Reuses cycle 245's CompressedMLP, fit_block_lstsq, eval_ppl, run_seed loop verbatim — only the outer SUBSETS loop and verdict block are new.
## Files
- NEW `experiments/four_block_sweep_cycle255.py`
- NEW `autoresearch/results/cycle_255_four_block_sweep.json`
- NEW `autoresearch/results/cycle_255_summary.md`
## Constraints
- SUBSETS=[[1,2,4,7],[1,4,7,8],[1,2,7,8]]; K=512; N_CALIB=200; SEQ_LEN=128; LR=1e-4; N_STEPS=4000; EVAL_EVERY=100
- SEEDS=[42,137,2024] per subset (9 fits total); fall back to SEEDS=[42,137] if wall time exceeds 30 min after the first 6 fits
- Same disjoint calib=all_ids[:200]/eval=all_ids[200:300] as cycle 245
- topk+lstsq init → joint Adam W_out+b_out; best-state restore; assert within 1e-3
- Stamp `cycle=255, item=265`; do NOT overwrite cycle 245 artifacts
## Verdict logic
Per item 265: CONFIRMED if min_over_subsets(best_mean) ≤ 51.0; REFUTED if all subsets best_mean ≥ 53.0; INCONCLUSIVE in (51.0, 53.0). best_mean is computed across all completed seeds per subset.
## Steps
1. Copy cycle 245 script body; set BLOCKS as a per-iteration variable from SUBSETS list.
2. For subset in SUBSETS: for seed in SEEDS: run_seed(seed) → record subset_results[subset][seed].
3. Aggregate per-subset best_mean ± std across all 3 seeds; apply verdict logic over the 3 subsets; write JSON + markdown summary.
4. Do not execute — runner schedules.
## Risks
3 subsets × 3 seeds = 9 fits ≈ 27 min total at ~3 min/fit — at the edge of the 30-min cap. Fallback: drop to 2 seeds if budget tight.
## Verification
`python -c "import ast; ast.parse(open('experiments/four_block_sweep_cycle255.py').read())"`; SUBSETS list matches; K==512; N_STEPS==4000; OUTPUT_PATH does not collide with cycle 245.
