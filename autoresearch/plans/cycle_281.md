# Cycle 281 Plan — Item 291: 2-block [1, 2] K=512

## Item (verbatim)
291 (P0): 2-block [1, 2] K=512 — does adding block 2 (lowest eff_rank) beat cycle 266 [1,4]?

## Hypothesis (item 291 verbatim)
Block 2's extreme rank concentration (cycle 278 eff_rank=1.05, top-1 energy 99.61%) means it contributes almost nothing new to the residual stream beyond what block 1 already contributes — adding it to [1] should either match cycle 266 [1,4] within noise (confirming "add low-eff_rank blocks for cheap") or improve on it (confirming "block 2 is on the load-bearing path and cycle 266 missed it").

## Setup
New script `experiments/two_block_1_2_cycle281.py`. Trivial SUBSETS swap on cycle 266 (`experiments/two_block_sweep_cycle266.py`) — set `SUBSETS = [[1, 2]]`, K=512, W_out+b_out trainable per block (cycle 266 form, NOT cycle 279 W_fc variant). CompressedMLP wrapper, fit_block_lstsq topk+lstsq init unchanged. Joint Adam LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore (W_out, b_out per block). 3 seeds {42, 137, 2024}, disjoint N_CALIB=200 / N_EVAL=100, SEQ_LEN=128, MPS, GPT-2 small wikitext-2 val. Trainable per fit ≈ 787,968 (= 2 × (512×768 + 768)). ~5 min/fit × 3 ≈ 15 min wall.

## Verdict thresholds (item 291 verbatim)
- CONFIRMED iff best_mean ≤ 43.9 (matches/beats cycle 266 [1,4]=43.852 within noise).
- REFUTED iff best_mean ≥ 44.5.
- INCONCLUSIVE otherwise.

## Steps
1. Create `experiments/two_block_1_2_cycle281.py` from cycle 266 template: SUBSETS=[[1,2]], OUTPUT_PATH→cycle_281, SUMMARY_PATH→cycle_281, item/cycle metadata→281/291, verdict thresholds rewritten per item 291, drop wall-time-cap fallback (single subset, ~15 min).
2. Run `MPLCONFIGDIR=.mplconfig .venv/bin/python experiments/two_block_1_2_cycle281.py 2>&1 | tee autoresearch/logs/cycle_281_run.log`.
3. Save `autoresearch/results/cycle_281_two_block_1_2.json` with per-seed history, aggregate (best_mean±std, best_step, init, final@4000), verdict + verdict_logic, config, baselines: cycle 266 [1,4]=43.852, cycle 271 [1]=43.821, float=53.895.
4. Save `autoresearch/results/cycle_281_summary.md` with verdict, per-seed table, aggregate, deltas vs cycle 266 [1,4], cycle 271 [1], float.

## Bookkeeping
Delete item 291 from NEXT/P0; insert `### Cycle 281` at top of PAST EXPERIMENTS (above cycle 280). Append TSV row to `autoresearch/logs/experiments.tsv` (HASH placeholder, update after commit). Stage ONLY cycle 281 files. Primary commit `[autoresearch] cycle 281: 2-block [1,2] K=512 — <verdict>`, capture hash, update TSV hash, write `autoresearch/results/cycle_281_manifest.txt`, bookkeeping commit. Write `autoresearch/results/cycle_281_exec_result.txt`. Audit: `autoresearch/audits/cycle_281.md` + `autoresearch/results/cycle_281_audit.txt`.

## Verification
JSON+log present; per-seed best_ppl, best_step, init_ppl, history recorded; restore matches recorded best (script assert <1e-3); init PPL deterministic across seeds (lstsq init seed-independent on identical calib); trainable param count asserted = 787,968 = 2×(512×768 + 768); verdict matches verbatim threshold logic.
