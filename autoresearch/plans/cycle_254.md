# Cycle 254 — Item 264: 5-block [1,2,4,7,8] K=128, 8000 steps
## Hypothesis
> extending the budget closes most of the residual gap → best_mean ≤ 56.0 (within 2.1 PPL of float, beats cycle 251 by ≥3 PPL, and well below cycle 231 K=128 6-block 67.562).
## Approach
Trivial extension of cycle 251. Copy `experiments/five_block_k128_cycle251.py` → `experiments/five_block_k128_8k_cycle254.py`. Only changes: `N_STEPS=8000`, `EVAL_EVERY=200`. Same BLOCKS, K, LR, seeds, calib/eval split.
## Files
- NEW `experiments/five_block_k128_8k_cycle254.py`
- NEW `autoresearch/results/cycle_254_five_block_k128_8k.json`
- NEW `autoresearch/results/cycle_254_summary.md`
## Constraints
- BLOCKS=[1,2,4,7,8]; K=128; LR=1e-4; SEEDS=[42,137,2024]; N_STEPS=8000; EVAL_EVERY=200
- topk+lstsq init → joint Adam W_out+b_out; best-state restore; assert within 1e-3
- Stamp `cycle=254, item=264`; rename `final_ppl_at_step_4000` → `final_ppl_at_step_8000` everywhere
- Do NOT overwrite cycle 251 artifacts
## Verdict logic
CONFIRMED ≤56.0; REFUTED ≥58.5; INCONCLUSIVE (56.0, 58.5). Anchors: cycle 251 59.119, cycle 245 K=512 53.826, cycle 231 K=128 6-block 67.562, float 53.895.
## Steps
1. Copy cycle 251 script; set `N_STEPS=8000`, `EVAL_EVERY=200`.
2. Update OUTPUT_PATH/SUMMARY_PATH; stamp `cycle=254, item=264`; rewrite verdict thresholds.
3. Rename `final_ppl_at_step_4000` field/prints to `final_ppl_at_step_8000`.
4. Do not execute — runner will schedule.
## Risks
5-block K=128 fits trivially in memory. Risk is wall time (~15 min). Best may still be at terminal step → INCONCLUSIVE plateau detection.
## Verification
`python -c "import ast; ast.parse(open('experiments/five_block_k128_8k_cycle254.py').read())"`; `N_STEPS==8000`, `EVAL_EVERY==200`, BLOCKS/K unchanged; OUTPUT_PATH does not collide with cycle 251.
