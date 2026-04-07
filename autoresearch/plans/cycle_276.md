# Cycle 276 Plan — Item 286: 1-block [1] K sweep — K ∈ {32, 64, 128, 256}

## Item (verbatim)
286 (P0): 1-block [1] K sweep — K ∈ {32, 64, 128, 256}. Find the 1-block floor.

## Hypothesis
1-block concave-up K-curve generalises but is SHALLOWER than multi-block → K=256 ≤ 45.5, K=128 ≤ 47, K=64 ≤ 50, K=32 ≤ 54; at least one K ∈ {32, 64, 128, 256} enters the Pareto frontier.

## Verdict thresholds (item 286 verbatim)
- CONFIRMED iff K=256 best_mean ≤ 45.5 OR K=128 best_mean ≤ 47.0.
- REFUTED iff ALL of {K=32, K=64, K=128, K=256} best_mean ≥ 50.0.
- INCONCLUSIVE otherwise.

## Setup
New script `experiments/one_block_k_sweep_cycle276.py` adapted from cycle 271 template (verbatim CompressedMLP / collect_mlp_io / fit_block_lstsq / eval_ppl). SUBSETS=[[1]] fixed; K_VALUES=[32, 64, 128, 256] is the sweep var. Per (K, seed): fresh GPT-2 small load, collect block 1 MLP I/O, install topk+lstsq CompressedMLP at block 1, freeze all params, unfreeze block 1 W_out (K×768) + b_out (768) JOINT Adam LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore (snapshot W_out+b_out), assert |restored − best| < 1e-3. 3 seeds {42, 137, 2024}. Disjoint N_CALIB=200/N_EVAL=100, SEQ_LEN=128, MPS. **12 fits total = 4 K × 3 seeds**. Trainable per K: K=32→25,344, K=64→49,920, K=128→99,072, K=256→197,376. Drop seed 2024 if wall-time cap (~30 min) hit after first 8 fits.

## Steps
1. Create `experiments/one_block_k_sweep_cycle276.py`. Loop outer over K, inner over seed. Track wall time; drop seed 2024 if cap exceeded after 8 fits.
2. Run `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/one_block_k_sweep_cycle276.py 2>&1 | tee autoresearch/logs/cycle_276_run.log`.
3. Save JSON `autoresearch/results/cycle_276_one_block_k_sweep.json` with all per-(K,seed) {init, best, best_step, final@4000, history}, per-K aggregates {best_mean, best_std, best_min, best_max, init_mean, final_mean}, baselines {cycle 271 [1] K=512=43.821, cycle 272 K=256 4-block min=51.922, cycle 268 K=64 4-block min=55.953, cycle 230 K=1 3-block=59.245, float=53.895}, verdict + verdict_logic.
4. Save markdown summary `autoresearch/results/cycle_276_summary.md`.

## Baselines
cycle 271 [1] K=512=43.821 (394K, current 1-block Pareto winner); cycle 272 K=256 4-block [1,2,4,7]=51.922 (789K); cycle 268 K=64 4-block [1,4,7,8]=55.953 (197K); cycle 230 K=1 3-block [1,4,7]=59.245 (5.8K); float 53.895.

## Bookkeeping
Delete item 286 from NEXT/P0 (item 287 becomes new top of P0); insert `### Cycle 276` at top of PAST EXPERIMENTS with verdict + Pareto reading. Append TSV row to `autoresearch/logs/experiments.tsv`. Stage cycle files only. Primary commit, get hash, manifest, bookkeeping commit. Audit at `autoresearch/audits/cycle_276.md` + `autoresearch/results/cycle_276_audit.txt`.
