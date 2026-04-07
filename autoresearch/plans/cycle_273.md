# Cycle 273 Plan — Item 283: eff_rank-guided heterogeneous K

## Hypothesis (verbatim)
Heterogeneous K guided by cycle 267 eff_rank — block 11 at K=22, blocks 1/4/7 at K=512. Test BLOCKS=[1,4,7,11] with K_PER_BLOCK={1:512, 4:512, 7:512, 11:22}. Plus control uniform K=22.

## Verdict thresholds
- CONFIRMED: hetero best_mean ≤ 44.5 AND control best_mean ≥ 60.
- REFUTED: hetero best_mean ≥ 47.
- INCONCLUSIVE: otherwise.

## Setup
Clone `experiments/hetero_k_6block_cycle247.py` → `experiments/hetero_k_eff_rank_cycle273.py`. BLOCKS=[1,4,7,11]. CONFIGS=[{"name":"hetero","k_per_block":{1:512,4:512,7:512,11:22}}, {"name":"uniform_k22","k_per_block":{1:22,4:22,7:22,11:22}}]. Joint Adam W_out+b_out, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024]. Disjoint calib (200) / eval (100) splits, SEQ_LEN=128. Best-state snapshot/restore, assert restore<1e-3. 6 fits total.

## Steps
1. Create `experiments/hetero_k_eff_rank_cycle273.py` from cycle 247 template. Replace BLOCKS, replace single K_PER_BLOCK with CONFIGS list. Wrap `run_seed` to take `(seed, k_per_block, cfg_name)` so EXPECTED_TRAINABLE recomputes per config. Hetero trainable = 3*(512*768+768)+1*(22*768+768) = 1,198,080. Uniform_k22 trainable = 4*(22*768+768) = 70,656.
2. Outer loop: for cfg in CONFIGS, for seed in SEEDS, call run_seed → list of {cfg_name, seed, init, best, best_step, final, history, per_block_init_mse}.
3. Aggregate per cfg: best_mean/std, init_mean, final_mean. Apply verdict on hetero best_mean (and control gate ≥60).
4. Save `autoresearch/results/cycle_273_hetero_eff_rank.json` (config block, all seeds, per-cfg aggregate, baselines: cycle266 [1,4]=43.852, cycle270 single-block-11 K=22=54.469, float=53.895, verdict).
5. Write `autoresearch/results/cycle_273_summary.md` (verdict header, per-cfg table, deltas vs baselines, trajectory snippets).
6. Run `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/hetero_k_eff_rank_cycle273.py 2>&1 | tee autoresearch/logs/cycle_273_run.log`.

## Baselines
cycle266 [1,4] K=512=43.852; cycle270 single-block-11 K=22=54.469; float 53.895; cycle247 hetero(6blk)=64.45.

## Bookkeeping
Update agenda.md: delete item 283 from NEXT/P1; add `### Cycle 273` to PAST EXPERIMENTS top with verdict. Append TSV row. Stage cycle files only. Verdict commit + bookkeeping commit + manifest. Audit.

## Expected outcome
If block 11 is rank-bottlenecked (eff_rank≈22), allocating K=22 there should match K=512 quality at much lower cost; cycle 270 single-block patch already hit 54.469. Three K=512 blocks at [1,4,7] alone (cycle 210) gave 46.56; adding rank-22 block 11 patch expected ~44-45.
