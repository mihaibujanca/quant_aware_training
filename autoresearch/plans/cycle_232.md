# Task
Cycle 232 / item 242: implement and run the sequential-init + joint-polish hybrid at BLOCKS=[1,4,7], K=512, comparing against the existing sequential-only (cycle 215, 47.253) and joint-only (cycle 206/210, 46.556) baselines. Test whether sequential warm-start gives joint Adam a better basin.

# Hypothesis
"Sequential init + joint polish hybrid at 3 blocks — break the 46.5 floor? Cycle 215 (sequential): 47.253. Cycle 206 (joint): 46.556. Cycle 215's audit recommended 'sequential + lightweight joint polish at end' as an untested composition. Implement: run cycle 215's sequential protocol (2000 steps per block with best-state restore) to convergence, then initialize a joint Adam pass on all 3 blocks' W_out+b_out simultaneously for 1000 more joint steps with LR=5e-5 (lower LR to avoid wiping the sequential solution). BLOCKS=[1,4,7] K=512, 3 seeds (42, 137, 2024). Compare final_mean and best_mean against both baselines. Tests whether sequential provides a better basin of attraction that joint polishing can exploit."

Confirmed if hybrid best_mean ≤ 45.8. Refuted if ≥ 46.5. Inconclusive otherwise.

# Pre-flight Results
- P1 (item exists in NEXT/P0): clean — agenda.md line 9.
- P2 (no prior cycle 232 artifacts): clean.
- P3 (dependencies done): cycle 215 (item 225) and cycle 206 (item 217) both completed.
- P4 (novel composition): clean — sequential alone tested in 215, joint alone tested in 206/210/213, never combined.

# Steps
1. Create `experiments/sequential_polish_hybrid_cycle232.py`.
2. Reuse cycle 215 helpers verbatim: `CompressedMLP`, `collect_mlp_io`, `fit_block_lstsq`, `eval_ppl`, calibration/eval loaders. Same SEQ_LEN=128, N_CALIB=200, N_EVAL=100.
3. Config: `BLOCKS=[1,4,7]`, `K=512`, `SEEDS=[42,137,2024]`, `LR_PHASE1=1e-4`, `LR_PHASE2=5e-5`, `N_STEPS_PHASE1=2000`, `N_STEPS_PHASE2=1000`, `EVAL_EVERY=100`.
4. Per seed:
   a. Build topk+lstsq compressed model on the 3 blocks (cycle 215 protocol). Record `init_ppl`.
   b. PHASE 1 — sequential (cycle 215 protocol, identical):
      - For `block_idx in [1, 4, 7]`:
        - Freeze all params, then unfreeze ONLY `mlp.W_out` and `mlp.b_out` for that block.
        - Snapshot pre-block state for collapse rollback.
        - Fresh `Adam([W_out, b_out], lr=LR_PHASE1)`.
        - Loop calibration shuffle for `N_STEPS_PHASE1` steps; eval every `EVAL_EVERY`; track best (W_out, b_out); restore best at end (or pre-block on collapse).
      - Record `post_sequential_ppl` (should match cycle 215 ≈ 47.25).
   c. PHASE 2 — joint polish:
      - Freeze all params; unfreeze ALL 3 blocks' `W_out` and `b_out` simultaneously (W_fc stays frozen).
      - Fresh `Adam(trainables, lr=LR_PHASE2)` (5e-5, lower than phase 1).
      - Snapshot pre-phase2 state (= phase-1 best) for collapse rollback.
      - Loop for `N_STEPS_PHASE2`; eval every `EVAL_EVERY`; track best `(W_out_b, b_out_b)` for all 3 blocks jointly; restore best at end.
      - Record `final_ppl`, `best_step`, `phase2_history`.
5. Aggregate (mean/std across 3 seeds): `init_mean`, `post_sequential_mean`, `final_mean`, `final_std`, `gain_from_polish_mean = post_sequential_mean - final_mean`.
6. Verdict per item 242: CONFIRMED if `final_mean ≤ 45.8`, REFUTED if `final_mean ≥ 46.5`, else INCONCLUSIVE.
7. Print summary table comparing: float, topk+lstsq init (61.746), sequential alone c215 (47.253), joint alone 1k c206 (48.111), joint alone 4k c210 (46.556), W_out+W_fc joint c213 (46.424), THIS hybrid (final_mean).
8. Save JSON to `autoresearch/results/cycle_232_sequential_polish.json` and markdown summary to `autoresearch/results/cycle_232_summary.md`.

# Validation
- `post_sequential_ppl_mean` ≈ 47.25 ± 0.1 (sanity-checks phase 1 reproduces cycle 215).
- `final_ppl < post_sequential_ppl` per seed (joint polish should add value, not destroy it).
- 3 seeds with `final_std < 0.5` (low variance, comparable to cycle 215's 0.047).
- Trainable count phase 1 = 768·K + 768 per block; phase 2 = 3·(768·K + 768).

# Artifacts
- `experiments/sequential_polish_hybrid_cycle232.py`
- `autoresearch/results/cycle_232_sequential_polish.json`
- `autoresearch/results/cycle_232_summary.md`
- `autoresearch/results/cycle_232_manifest.txt` (written by execute step)

# Fallbacks
- If wall time tight: `N_STEPS_PHASE1=1500` per block, `N_STEPS_PHASE2=800`. Document deviation.
- If phase 2 makes it WORSE per seed (`final > post_sequential`): record honestly, do NOT restore phase-1 state silently — report it as evidence the lower-LR joint cannot improve on sequential.
- If phase 2 collapses (best > post_sequential): rollback to phase-1 state, mark `phase2_collapsed=True`, still report.

# Update bookkeeping
- Move item 242 from `agenda.md` NEXT/P0 to PAST/Cycle 232 with one-line verdict.
- Append TSV row to `autoresearch/cycles.tsv`: cycle 232, item 242, verdict, final_mean, final_std.
- Update `autoresearch/STATUS.md` with cycle 232 entry.
- Commit (separate from execute commit) once results land.
