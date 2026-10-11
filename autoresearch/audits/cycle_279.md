# Cycle 279 Audit

**Status: PASS**

## Item
289 (P2): Cycle 266 [1,4] K=512 + W_out + W_fc trainable — does unfreezing W_fc help at the 2-block Pareto winner?

## Verdict
CONFIRMED (W_fc noop). best_mean = 43.87751 ± 0.09921 lands squarely in the CONFIRMED band (43.6, 44.1) per item 289 verbatim thresholds (margin 0.278 from lower bound, 0.222 from upper bound). +0.026 PPL vs cycle 266 [1,4] W_out-only baseline (43.852) — within noise.

## Verification

### Plan adherence
- Plan: new script `experiments/two_block_wout_wfc_cycle279.py` combining cycle 266 2-block scaffold with cycle 213 FullCompressedMLP (SLICED W_fc semantics). PASS
- Script sets BLOCKS=[1, 4], K=512, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEEDS=[42, 137, 2024], N_CALIB=200, N_EVAL=100, SEQ_LEN=128, DEVICE=mps (lines 29-43). PASS
- `FullCompressedMLP` verbatim from cycle 213 (W_fc Parameter + b_fc Parameter + W_out Parameter + b_out Parameter, forward = F.gelu(x @ W_fc + b_fc) @ W_out.T + b_out). PASS
- `fit_block_lstsq` verbatim from cycle 213 (topk by `act.std(0) * W_proj.norm(dim=1)`, lstsq init for W_out/b_out, W_fc/b_fc init = c_fc weight slice at topk_idx). PASS
- Adam unfreezes all 4 tensors per block: W_fc, b_fc, W_out, b_out. PASS
- Best-state restore snapshots all 4 tensors (not just W_out/b_out). PASS
- Disjoint calib/eval splits: `calib_ids = all_ids[:200]`, `eval_ids = all_ids[200:300]`. PASS
- 3 seeds {42, 137, 2024}. PASS
- Restore sanity assert `abs(final_ppl_restored - best_ppl) < 1e-3` per seed. All 3 pass per run log. PASS

### Sliced-vs-full discrepancy documentation
- Plan explicitly calls out the agenda setup line discrepancy (agenda claims W_fc_l (768×3072) → 5,512,704 total; plan/script/manifest/audit/commit use SLICED form → 1,575,424 total). PASS
- Script docstring explicitly flags the discrepancy (lines 1-15). PASS
- JSON output includes `sliced_semantics_note` field with full explanation. PASS
- Manifest includes `== SLICED-vs-full semantics (critical) ==` section. PASS
- Summary markdown includes `## SLICED-vs-full semantics note` section. PASS
- Agenda past-experiments entry notes the discrepancy in the outcome text. PASS
- TSV verdict_compact field notes sliced semantics. PASS
- Commit message notes the discrepancy. PASS
- Exec_result notes the discrepancy. PASS

### Numerical consistency
- Init PPL deterministic across all 3 seeds (57.05724) — correct, lstsq init is seed-independent for identical calib (topk idx + lstsq solve are both deterministic). PASS
- Float PPL = 53.895 exact match to reference (line logged). PASS
- Per-seed best: seed 42 = 44.0172 @ step 1700, seed 137 = 43.7961 @ step 2100, seed 2024 = 43.8192 @ step 2000. All within the hypothesized ±0.25 band around cycle 266 43.852. PASS
- Aggregate best_mean = (44.0172 + 43.7961 + 43.8192) / 3 = 43.87750 ≈ 43.87751 (matches JSON to 5 dp). PASS
- Aggregate best_std (population) = sqrt((0.1397² + 0.0814² + 0.0583²)/3) ≈ 0.0992 (matches JSON 0.09921). PASS
- Final@4000 mean = (46.9585 + 46.5842 + 46.3459)/3 = 46.6295 ≈ 46.62954 (matches JSON). PASS
- best_step_mean = (1700 + 2100 + 2000)/3 = 1933.33 ≈ 1933.3 (matches JSON). PASS
- Trainable params = 2 × (768×512 + 512 + 512×768 + 768) = 2 × (393,216 + 512 + 393,216 + 768) = 2 × 787,712 = 1,575,424 (script stdout prints "Trainable params: 1,575,424 (1.58M)" per seed, all 3 match). PASS
- Wall time 457.6 sec = 7.6 min (JSON `wall_time_sec`). Plan estimate 15 min was conservative. PASS

### Verdict logic
- Per item 289 verbatim: CONFIRMED iff best_mean ∈ (43.6, 44.1); REFUTED iff best_mean ≤ 43.5; INCONCLUSIVE if best_mean ∈ (43.5, 43.6) OR ≥ 44.1.
- best_mean = 43.87751
- 43.6 < 43.87751 < 44.1? TRUE (0.278 margin from lower, 0.222 margin from upper).
- → CONFIRMED matches JSON `verdict` field, script stdout, agenda entry, commit message, TSV row, manifest, and exec_result. PASS
- REFUTED clause: 43.87751 ≤ 43.5? FALSE (0.378 PPL above threshold). PASS
- INCONCLUSIVE clauses: 43.5 < 43.87751 < 43.6? FALSE (above 43.6). best_mean ≥ 44.1? FALSE (below 44.1 by 0.222). Neither INCONCLUSIVE clause triggers. PASS
- Note: the script verdict logic also catches the ≥ 44.1 regression branch as INCONCLUSIVE (not REFUTED), per item 289 semantics — "REFUTED" is reserved for the W_fc-helps direction (best_mean ≤ 43.5). Correct.

### Trainable count
- 1,575,424 trainable params (SLICED form). Matches plan, script docstring, script runtime output, JSON, summary, manifest, agenda entry, commit message.
- This is 2× cycle 266 [1,4] W_out-only (788K), not 7× as the agenda setup line claimed (5,512,704 for full c_fc interpretation).
- Sliced form is correct for apples-to-apples comparison with cycles 213/241/248/262. PASS

### Baseline cross-checks
- Cycle 266 [1,4] K=512 W_out-only baseline = 43.852 (from cycle 266 aggregate best_mean for subset [1,4]). Cycle 279 best_mean = 43.878 → delta = +0.026 PPL. Within ±0.25 apples-to-apples band per hypothesis. PASS
- Cycle 213 [1,4,7] K=512 W_out+W_fc baseline = 46.424 (from cycle 213 result, sliced semantics, early-stopping variant). Cycle 279 best_mean = 43.878 → delta = −2.546 PPL (2-block recipe beats 3-block variant in same W_out+W_fc family). PASS
- Cycle 248 6-block K=128 W_out+W_fc baseline = 67.609 (used for context only — confirms 6-block K=128 is far above 2-block K=512). PASS
- Cycle 271 [1] K=512 W_out-only baseline = 43.821 (from cycle 271 subset [1] best_mean). Cycle 279 best_mean = 43.878 → delta = +0.057 PPL (cycle 271 strictly dominates cycle 279 on both axes). PASS
- Float baseline 53.895 matches run (53.895). Cycle 279 delta vs float = −10.017. PASS

### Pareto check
- Cycle 279 point: (1,575,424 params, 43.878 PPL).
- Existing Pareto frontier: cycle 271 [1] K=512 (393,984, 43.821), cycle 266 [1,4] K=512 (788K, 43.852) dominated by cycle 271, cycle 276 K=256 1-block (197,376, 45.385), cycle 276 K=128 1-block (99,072, 46.932), cycle 276 K=64 1-block (49,920, 48.045), cycle 276 K=32 1-block (25,344, 49.649), cycle 270 block 11 r22 (17,664, 54.469).
- Cycle 279 at (1.58M, 43.878) is STRICTLY DOMINATED by cycle 271 [1] K=512 (394K, 43.821) on BOTH axes: 4× the params AND 0.057 PPL worse. Cycle 279 does NOT enter the Pareto frontier. PASS
- Manifest Pareto update section correctly notes this dominance. PASS

### Bookkeeping
- agenda.md: item 289 deleted from NEXT/P2 (P2 now empty; P0/P1 already empty). PASS
- agenda.md: `### Cycle 279` block inserted at top of PAST EXPERIMENTS (above `### Cycle 278`). Includes verdict CONFIRMED (W_fc noop), sliced-vs-full discrepancy flagged in-line, 8 geometric reading subsections, per-seed values, deltas vs all 4 baselines, W_fc probe family closure statement, artifacts list. PASS
- TSV row appended to `autoresearch/logs/experiments.tsv`: cycle=279, item=289, type=empirical, outcome=confirmed, commit=4591c88, all 8 columns (verified via csv reader → ncols=8). PASS
- Hash placeholder replaced post-commit via python script: `HASH_PLACEHOLDER` → `4591c88`. PASS
- Manifest written: `autoresearch/results/cycle_279_manifest.txt` with verdict logic, aggregate + per-seed table, sliced-vs-full note, setup, baselines, deltas, geometric reading, wall time, 6-artifact SHA256, Pareto update, commit=4591c88. PASS
- exec_result written: `autoresearch/results/cycle_279_exec_result.txt` with commit=4591c88, summary block, per-seed, sliced semantics note, wall time, key findings, artifact list. PASS
- Primary commit `4591c88` with message `[autoresearch] cycle 279: [1,4] K=512 + W_fc retrain — CONFIRMED (W_fc noop)` and full body documenting the sliced-vs-full discrepancy. PASS
- Pre-commit hook fired once (fixed trailing whitespace on cycle_279_summary.md + end-of-file on cycle_279_summary.md/cycle_279_two_block_wout_wfc.json); re-staged and re-committed cleanly. PASS
- Bookkeeping commit `c1da686` with message `[autoresearch] cycle 279: bookkeeping — hash update` (TSV hash + manifest + exec_result). PASS

### Staging hygiene
- Primary commit staged ONLY cycle 279 files + agenda.md + TSV:
  - experiments/two_block_wout_wfc_cycle279.py (new)
  - autoresearch/plans/cycle_279.md (new)
  - autoresearch/briefs/cycle_279.md (new)
  - autoresearch/logs/cycle_279_run.log (new)
  - autoresearch/results/cycle_279_two_block_wout_wfc.json (new)
  - autoresearch/results/cycle_279_summary.md (new)
  - autoresearch/agenda.md (modified)
  - autoresearch/logs/experiments.tsv (modified)
- Pre-existing dirty files (autoresearch/program.md, autoresearch/results/cycle_170_manifest.txt, cycle_205_manifest.txt, cycle_255_manifest.txt, cycle_266_exec_result.txt, cycle_270_exec_result.txt, cycle_270_manifest.txt, autoresearch/surveys/literature_map.md, autoresearch/logs/flagged.jsonl, aleph/pullback_geometry.py, all untracked audit/brief/plan/log files from earlier cycles) left untouched. PASS
- Bookkeeping commit staged ONLY: TSV (modified), cycle_279_manifest.txt (new), cycle_279_exec_result.txt (new). PASS

## Issues found
None. Verdict logic, numerical consistency (per-seed agg checks all pass), sliced-vs-full discrepancy documented thoroughly across all artifacts, bookkeeping and staging hygiene all check out. CONFIRMED is the correct verdict per item 289 verbatim thresholds — best_mean 43.87751 sits squarely in the (43.6, 44.1) band with comfortable margins on both sides.

The one notable ask for this cycle was handling the agenda setup-line discrepancy (full c_fc vs sliced topk-columns). The user explicitly directed us to follow cycle 213/248 sliced semantics; the plan, script, manifest, audit, commit, TSV, and agenda entry all document the discrepancy and the choice, so future readers can file a separate full-c_fc variant if that interpretation becomes a priority.

## Key findings
- **W_fc trainable is a NOOP at the 2-block Pareto winner** (best_mean 43.878 vs cycle 266 W_out-only 43.852, +0.026 PPL within noise). The cycle 213/241/248/262 "W_out absorbs the DOF" pattern generalizes to 2 blocks.
- **W_fc retraining probe family now CLOSED** across 5 configs: (K=1, 3b), (K=128, 6b) ×2, (K=512, 3b), (K=512, 6b), (K=512, 2b). All 5 land within ±0.1 PPL of W_out-only baselines. The "W_out absorbs the DOF" pattern is universal for joint Adam on topk+lstsq-initialised compressed MLPs at K ∈ {1, 128, 512} and blocks ∈ {2, 3, 6}.
- **Best_step halves vs cycle 266**: W_out-only cycle 266 [1,4] best_step = 4000 (still descending, budget-bounded) vs cycle 279 W_out+W_fc best_step ≈ 1933 (mid-training optimum). Doubling trainable DOFs (788K → 1.58M) halves time-to-optimum but then the surface OVERSHOOTS rather than continuing to descend.
- **Final@4000 overshoot = +2.75 PPL above best**: cycle 279 final@4000 mean = 46.630 vs best 43.878. Without best-state restore, the W_fc DOF would LEAVE the recipe 2.78 PPL WORSE than cycle 266 W_out-only (which doesn't overshoot). Raw overfitting tendency is direct evidence that W_fc trainable does NOT absorb a real DOF — it opens extra directions Adam exploits to overfit the 200-calib training set.
- **2-block ceiling pinned at ~43.85 regardless of trainable DOFs**: cycle 266 [1,4] W_out-only 43.852, cycle 279 [1,4] W_out+W_fc SLICED 43.878, cycle 271 [1] W_out-only 43.821 — three recipes span 0.057 PPL. The [1,4]/[1] cluster is a local minimum of joint Adam on topk+lstsq init at K=512, and W_fc trainable has zero bearing on where that minimum sits.
- **Cycle 279 is strictly dominated on both axes by cycle 271 [1] K=512 W_out-only**: cycle 271 (394K, 43.821) has 25% of cycle 279's params AND is 0.057 PPL better. Cycle 279 does NOT enter the Pareto frontier.
- **Agenda item 289 setup line erroneously claimed W_fc_l (768×3072)** → 5,512,704 total params, repeating the cycle 248 mistake. Cycle 279 follows cycle 213 FullCompressedMLP sliced semantics (1,575,424 trainable) for apples-to-apples comparison. The full c_fc variant (5.5M trainable) is a separate experiment that can be filed as a new item if priority.
- **Sanity**: init PPL deterministic 57.05724 across all 3 seeds (lstsq init is seed-independent for identical calib). Restore sanity assert passes for all 3 seeds (|restored - best| < 1e-3).
- **Wall time 7.6 min**, 3/3 fits, no seed drop.

## Status
PASS
