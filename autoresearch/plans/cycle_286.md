# Cycle 286 Plan — Item 296

## Dependencies
None (cycle 271 template, cycle 283 baseline already on disk).

## Item (verbatim Hypothesis)
1-block [2] K=512 — does block 2 alone at maximum K beat cycle 271 [1] K=512 (43.82) on the per-block axis? Extrapolating cycle 283's per-K [2]-beats-[1] trend, [2] K=512 should land below cycle 271's 43.82 — potentially the first sub-43 1-block K=512 measurement and a new dominant Pareto winner on the PPL axis.

## Approach
Mirror `experiments/one_block_sweep_cycle271.py` verbatim with **SUBSETS=[[2]]** and K=512 — trivial single-subset swap. Joint Adam over `W_out` + `b_out` (cycle 271 form), 3 seeds {42, 137, 2024}, best-state restore. Same scorer (`act.std(0) * W_proj.norm(dim=1)`), same lstsq init.

## Files
- `experiments/block2_k512_cycle286.py` (new)
- `autoresearch/results/cycle_286_block2_k512.json`, `_summary.md`, `_manifest.txt`
- `autoresearch/logs/cycle_286_run.log`

## Config
SUBSETS=[[2]]; K=512; LR=1e-4; N_STEPS=4000; EVAL_EVERY=100; SEEDS=[42,137,2024]; N_CALIB=200; N_EVAL=100 disjoint; SEQ_LEN=128; MPS; GPT-2 small; wikitext-2 val. Trainable per fit = 512×768 + 768 = 393,984. Wall ~7 min (1 subset × 3 seeds, matches cycle 281 timing).

## Verification
- Float PPL within 0.5 of 53.895.
- Per-seed init PPL identical (deterministic topk lstsq init, seed-free).
- Trainable param count == 393,984 asserted.
- Per-seed best-state restore: assert `|restored - best| < 1e-3`.
- All other params `requires_grad_(False)`.

## Verdict logic (item 296 verbatim)
- CONFIRMED iff best_mean ≤ 43.5 (beats cycle 271 [1] K=512=43.82 by ≥0.32 PPL).
- REFUTED iff best_mean ≥ 44.5.
- INCONCLUSIVE otherwise.

## Baselines to record
float 53.895; cycle 271 [1] K=512 = 43.821 (393,984 params, current PPL Pareto winner); cycle 283 [2] K=64 = 47.511 (49,920 params); cycle 283 [2] K=32 = 47.897; cycle 283 [2] K=16 = 49.462; cycle 271 [4] K=512 = 44.63; cycle 271 [7] K=512 = 45.37.
