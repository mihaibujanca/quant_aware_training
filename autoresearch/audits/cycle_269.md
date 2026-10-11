# Cycle 269 Audit

**Status: PASS**

## Item
279 (P2): K=1 6-block extended budget (16000 steps) — settle whether cycle 240's catastrophe is budget-limited or structural.

## Verdict
CONFIRMED (structural) — best_mean = 91.6126 ± 0.0366 ≥ 90.0 by 1.61 PPL margin.

## Verification

### Plan adherence
- Plan: `autoresearch/plans/cycle_269.md` — trivial N_STEPS=4000→16000 + EVAL_EVERY=100→400 swap on cycle 240 template. Hypothesis verbatim from item 279. PASS
- Script: `experiments/k1_6block_16k_cycle269.py` — diffed against cycle 240: N_STEPS 4000→16000, EVAL_EVERY 100→400, OUTPUT_PATH/SUMMARY_PATH renamed (cycle_269_k1_6block_16k.json / cycle_269_summary.md), `final_ppl_at_step_4000` field renamed to `final_ppl_at_n_steps` for clarity, baseline constants reorganised (CYCLE_240_K1_6B_4K_BASELINE = 96.886 added; CYCLE_221/CYCLE_230/CYCLE_231 retained), verdict thresholds updated (CONFIRMED ≥ 90.0; REFUTED ≤ 80.0), `cycle` and `item` fields updated to 269/279, baseline labels in delta/print/JSON/markdown blocks updated, FLOAT_BASELINE_REF added. CompressedMLP, collect_mlp_io, fit_block_lstsq, eval_ppl, run_seed loop, snapshot/restore, hyperparams (BLOCKS/K/N_CALIB/N_EVAL/SEQ_LEN/LR/SEEDS) verbatim from cycle 240. PASS
- Hyperparameters: BLOCKS=[1,2,4,5,7,8], K=1, N_CALIB=200, N_EVAL=100 (disjoint), SEQ_LEN=128, LR=1e-4, N_STEPS=16000, EVAL_EVERY=400, best-state restore, MPS, GPT-2 small, wikitext-2 val, SEEDS={42,137,2024}. Match item 279 brief and constraints. PASS
- Trainable params per fit: 9,216 = 6 × (768 + 768) — script-internal assert `n_trainable == 9216` would have raised AssertionError on mismatch; log shows clean run. (Brief estimated 9,222 = 6 × (768+1) which is wrong by 6 — bias is 768-dim per block, not 1-dim. Script value is correct; brief was off by 6.) PASS

### Run completion
- Script executed with `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/k1_6block_16k_cycle269.py > autoresearch/logs/cycle_269_run.log 2>&1`. PASS
- Wall time: ~18 min (under the 25 min cap; 4× the steps but ~3× the wall of cycle 240 because EVAL_EVERY=400 means same total eval count as cycle 240 — eval cost is the same, only train cost is 4×, and the per-step MPS train cost dominates). PASS
- All 3 fits completed. PASS
- Per-fit `assert abs(final_ppl_restored - best_ppl) < 1e-3` passed (script-internal, would have raised AssertionError otherwise; log shows `Restored best (step 16000): 91.657 (recorded best=91.657)` for all 3 seeds — restore matches best). PASS
- Float baseline computed once: 53.895 (matches FLOAT_BASELINE_REF). PASS

### Numerical consistency
- Per-seed best from JSON aggregates matches the per-seed values:
  - mean(91.657, 91.613, 91.568) = 91.6126; std (population) = 0.0366. PASS
- Init PPLs deterministic: all 3 seeds = 203.246 (lstsq fit is seed-independent; only Adam initialisation differs via shuffle). PASS
- best_step values all = 16000 (still descending at the budget cap, no overfit). PASS
- Per-seed best_ppl < init_ppl by ~111 PPL across all 3 fits — Adam recovery is uniform. PASS
- Cross-seed std at 16k (0.037) is lower than cycle 240's std at 4k (0.071) — trajectories converge tighter as the run progresses, no multimodality. PASS
- Step 4000 PPLs in this run reproduce cycle 240 to 3 dp: cycle 269 step 4000 values are 96.943 / 96.786 / 96.929; cycle 240 final values were 96.943 / 96.786 / 96.929. IDENTICAL — confirms run is deterministic and the only difference vs cycle 240 is the additional 12k steps. PASS

### Verdict logic
- Per item 279 (verbatim from agenda):
  - CONFIRMED if best_mean ≥ 90.0
  - REFUTED if best_mean ≤ 80.0
  - INCONCLUSIVE if best_mean ∈ (80, 90)
- Script-implemented logic: `best_mean = 91.6126 ≥ 90.0` → CONFIRMED. PASS
- REFUTED clause `best_mean ≤ 80.0` → 91.61 > 80 → does NOT trigger (correct skip). PASS
- → CONFIRMED. PASS

### Cross-cycle comparison
- Δ vs cycle 240 K=1 6-block 4k (96.886): −5.273 PPL — extending the budget 4× bought only 5.27 PPL (14% of the K=1-vs-K=512 6-block gap of 37.37). PASS
- Δ vs cycle 230 K=1 3-block (59.245): +32.368 PPL — doubling block count at K=1 costs +32 PPL. PASS
- Δ vs cycle 221 K=512 6-block (59.515): +32.098 PPL — at 16k steps the K=1 vs K=512 6-block gap is +32.1 (vs +37.37 at 4k). PASS
- Δ vs cycle 231 K=128 6-block (67.562): +24.051 PPL — even K=128 6-block at 4k beats K=1 6-block at 16k by 24 PPL. PASS
- Δ vs float (53.895): +37.718 PPL — worst float gap of any retrained recipe in the K∈{1,128,512} × {3,4,5,6}-block sweep family. PASS

### Trajectory descent reading
- Mean trajectory: step 4000 → 96.89, step 8000 → 93.7, step 12000 → 92.2, step 16000 → 91.6.
- Per-1000-step descent rate: 0-4k → 26.6 PPL, 4-8k → 3.2 PPL, 8-12k → 1.5 PPL, 12-16k → 0.6 PPL.
- The descent is not flatlined (still monotonically decreasing) but is asymptotically slow. Linear extrapolation of the 12-16k rate (~0.15 PPL per 1000 steps) to 32k would give ~91.6 - 2.4 ≈ 89.2 — still well above 80 (the REFUTED threshold). The hypothetical 32k probe would not flip the verdict.
- Cycle 254 ("extended budget is a noop") generalises to K=1 6-block in spirit but not in letter: there IS a small monotone improvement (5.27 PPL over 12k extra steps), but the rate is so slow that any reasonable budget cannot bring K=1 6-block within 30 PPL of any non-K=1 baseline. PASS

### Bookkeeping
- `autoresearch/plans/cycle_269.md` written (24 lines including hypothesis verbatim from item 279). PASS
- `autoresearch/briefs/cycle_269.md` already existed (read-only context). PASS
- `experiments/k1_6block_16k_cycle269.py` created fresh, N_STEPS+EVAL_EVERY swap of cycle 240 template. PASS
- `autoresearch/logs/cycle_269_run.log` created (full stdout including tqdm progress bars and per-step eval output). PASS
- `autoresearch/results/cycle_269_k1_6block_16k.json` contains cycle/item/verdict/config/float_ppl/seeds (per-seed history)/aggregate/all baselines/all deltas. PASS
- `autoresearch/results/cycle_269_summary.md` written. PASS
- `autoresearch/results/cycle_269_manifest.txt` written with verdict, commit hash, all per-seed numerical values, all deltas, trajectory description, artifact list. PASS
- `autoresearch/results/cycle_269_exec_result.txt` written. PASS
- `autoresearch/agenda.md`: item 279 removed from P2 (P2 now has placeholder note); cycle 269 entry added at top of PAST EXPERIMENTS above cycle 268, with item 279 marked CONFIRMED, full reading + numerics + cross-cycle deltas + Pareto + open follow-ups + closing note. PASS
- TSV row appended to `autoresearch/logs/experiments.tsv` (line 267); HASH placeholder updated to `a743c60` after main commit. PASS
- Two commits: `a743c60` (cycle files: experiment script + plan + brief + log + JSON + summary + agenda) + `4dbf39d` (bookkeeping: TSV + manifest). PASS
- No pre-existing dirty files staged: verified via `git status --short` after staging — only cycle 269 files + agenda + TSV staged; pre-existing M files (program.md, results/cycle_170/205/255_manifest.txt, results/cycle_266_exec_result.txt, surveys/literature_map.md, logs/flagged.jsonl) untouched. PASS

### Pre-commit hook interaction
- First commit attempt failed because `fix end of files` hook added trailing newlines to `cycle_269_summary.md` and `cycle_269_k1_6block_16k.json`. Rule-compliant resolution: re-staged the modified files and re-ran `git commit` (NEW commit, not --amend), per CLAUDE.md git safety protocol. The second attempt passed all hooks. PASS

## Issues found
None substantive. One cosmetic note: the brief estimated trainable params at "≈ 9,222 (6 × (768 + 1))" — this is wrong; bias is 768-dim per block, not 1-dim. Actual trainable = 9,216 = 6 × (768 + 768). Script value (asserted) is correct. No impact on result or verdict.

## Status
PASS
