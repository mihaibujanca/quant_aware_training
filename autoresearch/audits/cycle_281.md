# Cycle 281 Audit

**Status: PASS**

## Item
291 (P0): 2-block [1, 2] K=512 — does adding block 2 (lowest eff_rank, cycle 278) beat cycle 266 [1,4] (43.852)?

## Verdict
INCONCLUSIVE — best_mean = 44.3389 ± 0.0070 lands in the dead zone (43.9, 44.5). Fails CONFIRMED (≤43.9) by 0.439 PPL; fails REFUTED (≥44.5) by 0.161 PPL. Δ vs cycle 266 [1,4]=43.852: **+0.486 PPL WORSE**. Δ vs cycle 271 [1]=43.821: **+0.518 PPL WORSE** (strictly dominated on both trainable-param and PPL axes).

## Verification

### Plan adherence
- Plan: `autoresearch/plans/cycle_281.md` — 25 lines, 8 headings (Item, Hypothesis, Setup, Verdict thresholds, Steps, Bookkeeping, Verification) matching cycle 280 template. Hypothesis quoted verbatim from item 291 in `autoresearch/briefs/cycle_281.md`. PASS
- Script: `experiments/two_block_1_2_cycle281.py` — direct SUBSETS swap on cycle 266 template: `SUBSETS = [[1, 2]]`, K=512, CompressedMLP + fit_block_lstsq unchanged, wall-time-cap fallback removed (single subset = 3 fits, not 9). Verdict logic rewritten per item 291 verbatim (CONFIRMED/REFUTED on single-subset best_mean, not min-over-subsets). Trainable-count assertion `n_trainable == len(BLOCKS)*(K*768+768) == 787968`. PASS
- Hyperparameters: N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42,137,2024], best-state restore (W_out + b_out per block). Match item 291 brief and cycle 266 convention. PASS
- Trainable per fit: 787,968 = 2 × (512×768 + 768) = 2 × 394,368 — script-internal assert PASS for all 3 seeds. Matches brief exactly. PASS

### Run completion
- Script executed with `MPLCONFIGDIR=.mplconfig HF_HUB_DISABLE_PROGRESS_BARS=1 .venv/bin/python experiments/two_block_1_2_cycle281.py > autoresearch/logs/cycle_281_run.log 2>&1`. Exit 0. PASS
- HF progress bars disabled at script level via env var (learning from cycle 280's post-cleanup). Log is clean without post-processing. PASS
- Wall: 7.6 min total (3 fits × ~2.5 min each). Well under the ~15-min plan estimate. PASS
- All 3 seeds completed; per-seed restore PASSED (`|final_ppl_restored - best_ppl| < 1e-3`). PASS

### Numerical consistency
- Float PPL = 53.89489 — matches FLOAT_BASELINE_REF=53.895 to 4 dp. PASS
- Init PPL deterministic at 57.2068 across all 3 seeds (lstsq init is seed-independent on identical calib split; only Adam shuffle differs). PASS
- Init PPL 57.2068 is ~0.15 PPL higher than cycle 266 [1,4]'s 57.057 — consistent with replacing block 4 (eff_rank 49) with block 2 (eff_rank 1, less expressive), which hurts the pure no-retrain lstsq init slightly. PASS
- Per-seed best PPLs: seed 42 → 44.3473 @ step 3700, seed 137 → 44.3302 @ step 2800, seed 2024 → 44.3392 @ step 3400. best_mean = (44.3473 + 44.3302 + 44.3392) / 3 = 44.3389, std (population) = sqrt(((44.3473-44.3389)² + (44.3302-44.3389)² + (44.3392-44.3389)²)/3) ≈ 0.0070 — matches JSON aggregate {best_mean: 44.3389, best_std: 0.0070} to 4 dp. PASS
- best_step_mean = (3700 + 2800 + 3400) / 3 = 3300, std ≈ 374 — matches JSON. PASS
- Per-seed restored matches recorded best exactly to 4 dp (assert PASS from script logs). PASS
- best_std = 0.0070 is the TIGHTEST of any 2-block K=512 recipe tested: cycle 266 [1,4]=0.0273 (4× looser), [1,7]=0.0508, [4,7]=0.0164. Noted in reading. PASS

### Verdict logic
- Per item 291 (verbatim from agenda):
  - CONFIRMED iff best_mean ≤ 43.9
  - REFUTED iff best_mean ≥ 44.5
  - INCONCLUSIVE otherwise
- Script-implemented logic matches verbatim:
  - best_mean = 44.3389 ≤ 43.9? NO (misses by 0.439)
  - best_mean = 44.3389 ≥ 44.5? NO (misses by 0.161)
  - → INCONCLUSIVE. PASS
- Verdict_logic dict in JSON: `{criterion_confirmed: "<= 43.9", criterion_refuted: ">= 44.5", best_mean: 44.3389, le_confirmed: false, ge_refuted: false, verdict: "INCONCLUSIVE"}` — internally consistent. PASS

### Geometric / cross-cycle reading
- **Headline finding: block 2 is NOT a cheap cascade add**. Item 291's core hypothesis — that block 2's extreme rank concentration (cycle 278: eff_rank=1.05, top-1 energy 99.61%) would make it a "free add" to cycle 271's [1]-only recipe — is WRONG. Adding block 2 to [1] makes things strictly WORSE by 0.518 PPL, with 2× the trainable params. Block 2 is NOT on the load-bearing path that [1,4] / [1] exploit. PASS
- **Third independent refutation of activation eff_rank as prescriptive signal** for cascade compression. Priors: cycle 273 REFUTED eff_rank-guided heterogeneous K; cycle 280 showed activation eff_rank ≠ weight-SVD rank at single-block level. Cycle 281 now adds: activation eff_rank does not predict which 2-block subset is cheapest. Block 2 has eff_rank 25× lower than block 4 (1.05 vs 49.04) but [1,2] is 0.486 PPL WORSE than [1,4]. Activation eff_rank is descriptive of per-block energy concentration but has no prescriptive power for cascade compression under joint Adam retrain. PASS
- **Tightest best_std of any 2-block K=512 recipe**: cycle 281 [1,2] = 0.0070 vs cycle 266 [1,4] = 0.0273 (4× looser), [1,7] = 0.0508 (7× looser), [4,7] = 0.0164 (2× looser). Geometric reading: block 2's near-rank-1 operator shape (eff_rank 1.05) contributes essentially ONE direction of freedom to the joint optimization — both highly stable across seeds (tight std) and cheap (doesn't add capacity). This is a direct numerical confirmation of cycle 278's eff_rank measurement. Low eff_rank = flat surface = stable BUT also = no room to absorb joint Adam DOFs. PASS
- **Block 2 is dominated on both axes by cycle 271 [1]-alone**: cycle 271 [1] K=512 W_out-only = 43.821 at 393,984 trainable params; cycle 281 [1,2] K=512 W_out-only = 44.339 at 787,968 trainable params. Cycle 271 is strictly better: 50% fewer trainable params AND 0.518 PPL better. Adding block 2 is Pareto-dominated. PASS
- **The 1-block [1] ceiling is the real Pareto anchor at K=512**: cycle 271 [1] = 43.821 (394K params). Cycle 266 [1,4] = 43.852 (788K) is only −0.032 better but 2× the params, essentially matching within noise. Cycle 281 [1,2] = 44.339 (788K) is strictly worse. Cycle 266 [1,7] = 43.905 (788K) is slightly worse. Cycle 266 [4,7] = 45.770 (788K) is much worse. The K=512 2-block recipe space sorts as: [1,4] ≈ [1,7] << [1,2] < [4,7]; and [1]-alone is tied with the best 2-block. This is consistent with cycle 271's finding that the count-driven cascade FLATTENS at 1 block (the 2→1 step saved only 0.03 PPL). PASS
- **Final@4000 vs best_mean overshoot**: cycle 281 final@4000 = 44.4636 ± 0.0278, best_mean = 44.3389 ± 0.0070. Overshoot = +0.125 PPL — small and tight. Compare to cycle 279 [1,4] W_out+W_fc overshoot of +2.75 PPL. The W_out-only cycle 281 [1,2] surface is mildly overshooting in the final 1000 steps but not collapsing. best_step_mean = 3300 suggests the optimum lies around step 2800-3700, not at step 4000 — this is mid-training rather than budget-bounded, slightly different from cycle 266 [1,4] where best_step=4000 (still descending). Interpretation: block 2's minimal DOF means the [1,2] surface converges faster and flatter than [1,4]. PASS
- **Open: item 293 [2] K-sweep**: cycle 281 now sets a negative prior for block 2 in isolation. If block 2 contributes nothing new additively to [1], then [2]-alone is likely worse than [1]-alone at every K. But the single-block test is still the first measurement of block 2 as a compression target, so item 293 remains worth running. PASS
- **Cycle 281 beats float by 9.56 PPL**: 44.339 vs 53.895. In absolute terms the [1,2] recipe is a strong compression, just not better than [1]-alone or [1,4]. PASS

### Bookkeeping
- `autoresearch/plans/cycle_281.md` written (25 lines, 8 headings, hypothesis verbatim from item 291, exact procedure, verdict thresholds, baselines, artifacts list, verification checklist). PASS
- `autoresearch/briefs/cycle_281.md` pre-existing (read-only context from planning agent). PASS
- `experiments/two_block_1_2_cycle281.py` created from cycle 266 template with SUBSETS=[[1,2]] swap, verdict-logic rewrite per item 291, wall-time-cap fallback removed (single subset), trainable-count assert added. PASS
- `autoresearch/logs/cycle_281_run.log` created via `tee`. Clean (HF progress bars disabled via env var). PASS
- `autoresearch/results/cycle_281_two_block_1_2.json` contains cycle/item/verdict/verdict_logic/float_ppl/subset_results (per-seed history)/aggregates/best_mean/best_std/all baselines/deltas/wall_time. Verified. PASS
- `autoresearch/results/cycle_281_summary.md` written (aggregate table, per-seed table, comparison deltas). PASS
- `autoresearch/results/cycle_281_manifest.txt` written with verdict, both commit hashes, per-seed results, deltas, full geometric reading. PASS
- `autoresearch/results/cycle_281_exec_result.txt` written with both commit hashes. PASS
- `autoresearch/agenda.md`: item 291 removed from P0; cycle 281 entry added at top of PAST EXPERIMENTS (above cycle 280), with item 291 marked INCONCLUSIVE, full reading + numerics + cross-cycle deltas + Pareto status. PASS
- TSV row appended to `autoresearch/logs/experiments.tsv` (line 282); HASH placeholder updated to `a738a92` after main commit. PASS
- Two commits: `a738a92` (cycle files: experiment script + plan + brief + log + JSON + summary + agenda) + `d6eb577` (bookkeeping: TSV + manifest + exec_result). PASS
- No pre-existing dirty files staged: verified via `git status --short` — only cycle 281 files + agenda + TSV + manifest + exec_result staged; pre-existing M files (program.md, flagged.jsonl, cycle_170/205/255_manifest.txt, cycle_266/270_exec_result.txt, cycle_270_manifest.txt, surveys/literature_map.md, aleph/pullback_geometry.py + untracked audits/reviews/plans/briefs from prior cycles) untouched. PASS

### Pre-commit hook interaction
- First main-commit attempt failed because `fix end of files` hook added trailing newlines to `cycle_281_summary.md` and `cycle_281_two_block_1_2.json`. Rule-compliant resolution: re-staged the modified files and re-ran `git commit` (NEW commit, not --amend), per CLAUDE.md git safety protocol. Second attempt passed all hooks (commit a738a92). PASS
- Bookkeeping commit passed all hooks first try (d6eb577). PASS

## Issues found
1. **Hypothesis was wrong**. Item 291 bet that block 2's extreme rank concentration (cycle 278 eff_rank=1.05) would make it a cheap add to the cycle 271 [1] recipe. Actual result: [1,2] is 0.518 PPL WORSE than [1]-alone, strictly dominated on both axes. This is now the third independent refutation of "activation eff_rank is prescriptive for cascade compression" (after cycle 273 and cycle 280).
2. **Verdict sits in the narrow dead zone**. best_mean 44.339 is 0.439 above the CONFIRMED ceiling and 0.161 below the REFUTED floor. The verdict is INCONCLUSIVE in the formal sense, but the descriptive reading is clear and negative: block 2 is not on the cascade Pareto frontier. The 0.6 PPL dead zone in item 291's thresholds was calibrated for "match or improve on [1,4]" — narrower thresholds (e.g. CONFIRMED ≤ 43.9, REFUTED ≥ 44.1) would have put this result cleanly in REFUTED territory given cycle 266 [1,4]'s 0.027 best_std. Worth noting for future cascade-add items: the REFUTED threshold should be tighter when the baseline is known to have very small seed variance.
3. **Tight best_std (0.0070) is an unusual numerical signature** — 4× tighter than cycle 266 [1,4], 7× tighter than cycle 266 [1,7]. Not a bug, but worth flagging: extremely low activation eff_rank (block 2 = 1.05) correlates with extremely flat joint-Adam loss surface. This could be a general pattern — items 292 (block 10 rank-17) and 293 (block 2 K-sweep) should both record best_std and check whether low-eff_rank correlates with tight std.
4. **best_step_mean = 3300 ± 374** — not budget-bounded (best_step ≠ 4000 on any seed). Cycle 266 [1,4] had best_step=4000 on all seeds (still descending). Cycle 281 [1,2] converges faster and stops improving ~step 3300. Consistent with block 2 being a rank-1-ish operator that has very little capacity to absorb additional Adam DOFs. Not a defect.

## Status
PASS
