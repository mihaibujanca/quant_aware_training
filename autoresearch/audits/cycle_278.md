# Cycle 278 Audit

**Status: PASS**

## Item
288 (P1): All-blocks c_proj output eff_rank census — extend cycle 267 to blocks {0, 2, 3, 4, 6, 7, 8, 9, 10}. Pure analysis, 0 trainable params, no Adam, no retraining.

## Verdict
INCONCLUSIVE — clause 1 of CONFIRMED triggers (block 4 eff_rank = 49.0399 ≤ 50, by 0.9601 margin), clause 2 fails (mid-blocks {2, 3, 10} = {1.05, 24.29, 16.70} all far below 100), REFUTED clause fails (block 4 = 49.04 < 100). Per item 288 verbatim thresholds.

## Verification

### Plan adherence
- Plan: new script `experiments/all_blocks_eff_rank_cycle278.py` extending cycle 267 SVD machinery to all 12 blocks. Verbatim reuse of hook + covariance + eigh + eff_rank pipeline. Sanity check vs cycle 267 for blocks {1, 5, 11}. PASS
- Script sets BLOCKS = list(range(12)), N_CALIB=200, SEQ_LEN=128, SEED=42, DEVICE=mps (lines 26-36). PASS
- Single forward pass over 200 calib seqs, hooks on all 12 mlp.c_proj modules collected concurrently. PASS
- Per-block 768x768 covariance + torch.linalg.eigh + eff_rank = exp(H(p)) per cycle 267. PASS
- Per-block c_proj.weight (3072x768) SVD top-20 sv computed and saved. PASS
- Sanity check assertion `assert delta < 1e-3` for blocks {1, 5, 11} vs cycle 267 reference values. All 3 pass with delta = 0 (identical numerics). PASS
- 0 trainable params, no fits, no Adam — pure analysis as required by item 288. PASS
- Wall ~5 min on MPS as planned. PASS

### Numerical consistency
- Cycle 267 sanity reproduction (item 288 verbatim requirement):
  - block 1: cycle 267 = 1.9448049068450928, cycle 278 = 1.9448049068450928, delta = 0.0. PASS
  - block 5: cycle 267 = 179.1913299560547, cycle 278 = 179.1913299560547, delta = 0.0. PASS
  - block 11: cycle 267 = 22.488744735717773, cycle 278 = 22.488744735717773, delta = 0.0. PASS
- All 3 sanity checks pass with EXACT (zero) delta — same calib slice, deterministic forward pass on MPS, identical numerical pipeline. Better than the < 1e-3 threshold required.
- 12-block eff_rank profile (run log / JSON cross-check):
  - block 0: 71.9302
  - block 1: 1.9448 (sanity)
  - block 2: 1.0503
  - block 3: 24.2853
  - block 4: 49.0399
  - block 5: 179.1913 (sanity)
  - block 6: 290.9718
  - block 7: 297.7935
  - block 8: 291.4056
  - block 9: 195.4961
  - block 10: 16.7024
  - block 11: 22.4887 (sanity)
  - Run log values match JSON `per_block.{b}.eff_rank` for all 12 blocks within rounding. PASS
- Per-block top-1 energy at the heavy-rank-1 blocks:
  - block 1: 0.9294 (matches cycle 267 0.9293517470359802 within 1e-4)
  - block 2: 0.9961 (essentially rank-1 — top-1 alone captures 99.61%)
  - block 11: 0.4895 (matches cycle 267 0.4894711971282959)
  - PASS
- Per-block c_proj.weight top-1 sv (operator side) sanity for blocks {1, 5, 11}:
  - block 1: 28.7065 (matches cycle 267 28.706462860107422)
  - block 5: 27.6176 (matches cycle 267 27.617647171020508)
  - block 11: 60.9931 (matches cycle 267 60.99306106567383)
  - PASS

### Verdict logic
- Per item 288 verbatim: CONFIRMED iff (b4 ≤ 50 OR b7 ≤ 50) AND ALL of {2, 3, 6, 8, 9, 10} ≥ 100; REFUTED iff b4 AND b7 BOTH ≥ 100; INCONCLUSIVE otherwise.
- Block 4 = 49.0399 ≤ 50? TRUE (margin 0.9601). Clause 1 of CONFIRMED first conjunct TRIGGERS. PASS
- Block 7 = 297.7935 ≤ 50? FALSE. (clause 1 already triggered by b4)
- ALL of {2, 3, 6, 8, 9, 10} ≥ 100?
  - block 2: 1.0503 < 100 → FAIL
  - block 3: 24.2853 < 100 → FAIL
  - block 6: 290.9718 ≥ 100 → PASS
  - block 8: 291.4056 ≥ 100 → PASS
  - block 9: 195.4961 ≥ 100 → PASS
  - block 10: 16.7024 < 100 → FAIL
  - 3 of 6 mid-blocks BELOW 100 → CONFIRMED clause 2 FAILS. PASS
- → CONFIRMED requires BOTH clauses → fails (clause 2 false).
- REFUTED iff b4 AND b7 BOTH ≥ 100? b4 = 49.04 < 100 → FAILS regardless of b7. PASS
- Neither CONFIRMED nor REFUTED → **INCONCLUSIVE**. Verdict matches JSON `verdict` field, script stdout, agenda entry, commit message, TSV row, manifest, and exec_result. PASS

### Trainable count
- 0 trainable params (pure analysis). No fit, no Adam, no retraining. Item 288 explicitly requires "Trainable params: 0". PASS

### Baseline cross-checks
- Cycle 267 reference values for blocks {1, 5, 11} loaded as `CYCLE_267_EFF_RANK` constant in script (lines 41-45) and asserted to match within 1e-3. All match with delta = 0. PASS
- No other baselines required (pure analysis, no PPL).
- Float baseline 53.895 mentioned in agenda baselines but not used (no PPL evaluation in this cycle).

### Pareto check
- Cycle 278 produces 0 PPL points (pure analysis). No new Pareto points added. No existing Pareto points modified. PASS
- Implication: blocks 2 and 10 flagged as unexploited rank-concentrated single-block-patch candidates. These will only appear on the Pareto frontier when (and if) future cycles run cycle 270 analogues at blocks 10 and 2.

### Bookkeeping
- agenda.md: item 288 deleted from NEXT/P1 (P1 now empty; P0 stays empty; P2 starts with item 289). PASS
- agenda.md: `### Cycle 278` block inserted at top of PAST EXPERIMENTS (above `### Cycle 276` since cycle 277 was PLAN INVALID and never made it to PAST). Includes verdict INCONCLUSIVE, full 12-block eff_rank profile in tabular notation, 9 labelled subsections of geometric reading, sanity check note, artifacts list. PASS
- TSV row appended to `autoresearch/logs/experiments.tsv`: cycle=278, item=288, type=analysis, outcome=inconclusive, commit=f780cbd, all 8 columns (verified via awk -F'\t' → NF=8). Hash placeholder replaced post-commit via python script. PASS
- Manifest written: `autoresearch/results/cycle_278_manifest.txt` with verdict logic, full 12-block profile, sanity check, geometric reading, 6-artifact SHA256, commit=f780cbd. PASS
- exec_result written: `autoresearch/results/cycle_278_exec_result.txt` with commit=f780cbd, summary block, profile, key finding, artifact list. PASS
- Primary commit `f780cbd` with message `[autoresearch] cycle 278: all-blocks eff_rank census — INCONCLUSIVE`. PASS
- Pre-commit hook fired once (fixed end-of-file on cycle_278_all_blocks_eff_rank.json); re-staged and re-committed cleanly. PASS
- Bookkeeping commit `ee820af` with message `[autoresearch] cycle 278: bookkeeping — hash update` (TSV hash + manifest + exec_result). PASS

### Staging hygiene
- Primary commit staged ONLY cycle 278 files + agenda.md + TSV:
  - experiments/all_blocks_eff_rank_cycle278.py (new)
  - autoresearch/plans/cycle_278.md (new)
  - autoresearch/briefs/cycle_278.md (new)
  - autoresearch/logs/cycle_278_run.log (new)
  - autoresearch/results/cycle_278_all_blocks_eff_rank.json (new)
  - autoresearch/results/cycle_278_summary.md (new)
  - autoresearch/agenda.md (modified)
  - autoresearch/logs/experiments.tsv (modified)
- Pre-existing dirty files (autoresearch/program.md, autoresearch/results/cycle_170_manifest.txt, cycle_205_manifest.txt, cycle_255_manifest.txt, cycle_266_exec_result.txt, cycle_270_exec_result.txt, cycle_270_manifest.txt, autoresearch/surveys/literature_map.md, autoresearch/logs/flagged.jsonl, aleph/pullback_geometry.py, all untracked audit/brief/plan/log files from earlier cycles) left untouched. PASS
- Bookkeeping commit staged ONLY: TSV (modified), cycle_278_manifest.txt (new), cycle_278_exec_result.txt (new). PASS

## Issues found
None. Verdict logic, numerical consistency (sanity checks pass with delta = 0), bookkeeping, and staging hygiene all check out. INCONCLUSIVE is the correct verdict per item 288 verbatim thresholds — clause 1 of CONFIRMED triggers (block 4 = 49.04 ≤ 50 by 0.96 margin) but clause 2 fails (mid-blocks 2/3/10 all < 100). REFUTED clause also fails (block 4 < 100).

## Key findings
- **U-shape rank-concentration hypothesis is REFUTED at the descriptive level**. Actual pattern is bathtub-with-block-0-bump: 4 contiguous low-rank concentrators at blocks 1-4 (eff_rank 1.05-49.04) with block 0 intermediate (71.93), 5 contiguous high-rank diffusers at blocks 5-9 (179.19-297.79), 2 low-rank concentrators at the late edge blocks 10-11 (16.70, 22.49). The early-block low-rank cluster is bigger and lower than the agenda hypothesis predicted.
- **Block 2 is essentially rank-1**: eff_rank 1.05, top-1 energy 99.61% — even more concentrated than block 1 (eff_rank 1.94, top-1 energy 92.94%). Block 2 has never been individually tested as a compression target.
- **Block 10 is the surprise**: eff_rank 16.70 < block 11 (22.49). Block 10 is the most rank-concentrated late block in the model and is currently unexploited by any cycle. Cycle 270 only tested block 11; an analogous rank-16 weight-SVD patch at block 10 may be a parallel 17.6K-param Pareto candidate.
- **Block 4 is on the boundary** (49.04, just under the 50 threshold). Less starkly low-rank than blocks 10/11. The CONFIRMED clause 1 triggers but block 4 is borderline.
- **Block 7 is HIGH-rank** (297.79). The Pareto-winning [1,4,7] cascade includes ONE concentrator (block 1, rank 2), one boundary (block 4, rank 49), and one diffuser (block 7, rank 298) — the cascade does NOT pick blocks by activation eff_rank. Cycle 273 already REFUTED eff_rank-guided heterogeneous K at the cascade level; cycle 278 confirms activation eff_rank is descriptive of energy concentration but NOT prescriptive as a per-block K budget for joint Adam compression.
- **Eff_rank ordering vs cycle 271 1-block PPL ordering**: 1-block best_means [1]=43.82 < [4]=44.63 < [7]=45.37 are in the SAME direction as eff_rank ordering 1.94 < 49.04 < 297.79, so a weak monotonic correlation exists. But the 150× rank range maps to only 1.55 PPL spread.
- **Two unexploited candidates flagged**: block 2 (rank ~1, untested) and block 10 (rank ~17, untested). Both have lower eff_rank than block 11 (the only late-block rank patch tested).
- **Sanity checks reproduce cycle 267 with delta = 0**: blocks {1, 5, 11} eff_ranks identical to cycle 267 reference values (1.9448, 179.1913, 22.4887). Same calib slice, deterministic forward pass on MPS, identical numerical pipeline.
- **Wall ~5 min**, 0 trainable params, no Adam, no fits — purely descriptive analysis as required by item 288.

## Status
PASS
