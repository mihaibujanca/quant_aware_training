# Cycle 282 Plan — Item 292: Block 10 c_proj rank-17 weight-SVD

## Item (verbatim)
292 (P0): Block 10 c_proj rank-17 weight-SVD — extending cycle 270 to the second-lowest late-block eff_rank.

## Hypothesis (item 292 verbatim)
Cycle 278 shows block 10 c_proj output eff_rank = 16.70, LOWER than block 11 (22.49). Block 10 is the most rank-concentrated late block in the model and is currently unexploited by any cycle. Cycle 270 showed weight-SVD rank-22 on block 11 retrains to 54.47 PPL at 17.7K params (new Pareto anchor). Hypothesis: analogous rank-17 patch on block 10 reaches ≤55 PPL at ~13.1K params (768×17 + 17×768) — a new lowest-param Pareto anchor. Note cycle 270 showed activation eff_rank is a WEAK indicator of weight-SVD compressibility; block 10's true useful rank may be higher. Test rank sweep to locate it.

## Setup
New script `experiments/block10_lowrank_cycle282.py`. Adapt cycle 270 (`experiments/block11_lowrank_cycle270.py`) verbatim but with TARGET_BLOCK=10 and RANKS_NORETRAIN=[5, 17, 50]. Retrain at r=17: joint Adam LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, best-state restore (B + bias). 3 seeds {42, 137, 2024}. Trainable per fit = 17×768 + 768 = 13,824 params — asserted. Disjoint N_CALIB=200 / N_EVAL=100, SEQ_LEN=128, MPS, GPT-2 small wikitext-2 val. ~2 min/fit × (3 noretrain + 3 retrain) ≈ 12 min wall.

## Verdict thresholds (item 292 verbatim)
- CONFIRMED iff rank-17 retrained best_mean ≤ 55.0 (matches cycle 270 band within 0.5 PPL).
- REFUTED iff all ranks (r∈{5,17,50}) no-retrain AND rank-17 retrain ≥ 57.0.
- INCONCLUSIVE otherwise.

## Steps
1. Create `experiments/block10_lowrank_cycle282.py` from cycle 270 template: TARGET_BLOCK=10, RANKS_NORETRAIN=[5,17,50], RETRAIN_RANK=17, thresholds per item 292, JSON→cycle_282_block10_lowrank, summary→cycle_282_summary.
2. Run `MPLCONFIGDIR=.mplconfig HF_HUB_DISABLE_PROGRESS_BARS=1 .venv/bin/python experiments/block10_lowrank_cycle282.py 2>&1 | tee autoresearch/logs/cycle_282_run.log`.
3. Save `autoresearch/results/cycle_282_block10_lowrank.json` with float, noretrain PPLs (r=5,17,50), retrain 3-seed aggregate, baselines (cycle 270 block 11 r=22 retrain=54.47, cycle 280 block 1 r=2 retrain=49.032, float=53.895).
4. Save `autoresearch/results/cycle_282_summary.md` with verdict, noretrain table, retrain table, Pareto deltas.

## Bookkeeping
Delete item 292 from NEXT/P0; insert `### Cycle 282` at top of PAST EXPERIMENTS (above cycle 281). Append TSV row to `autoresearch/logs/experiments.tsv` (HASH placeholder, update after commit). Stage ONLY cycle 282 files. Primary commit `[autoresearch] cycle 282: block 10 rank-17 weight-SVD — <verdict>`, capture hash, update TSV hash, write `autoresearch/results/cycle_282_manifest.txt`, bookkeeping commit. Write `autoresearch/results/cycle_282_exec_result.txt`. Audit: `autoresearch/audits/cycle_282.md` + `autoresearch/results/cycle_282_audit.txt`.

## Verification
JSON+log present; per-seed best_ppl/best_step/init_ppl/history recorded; restore matches recorded best (<1e-3); init PPL deterministic across seeds; trainable asserted = 13,824 = 17×768 + 768; verdict matches verbatim threshold logic; SVD reproducible across seeds (same W → same U,S,Vh).
