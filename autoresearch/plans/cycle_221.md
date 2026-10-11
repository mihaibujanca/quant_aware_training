# Task
Cycle 221 — Item 226 (P0): Adam correction for 6 blocks with proper early stopping. Rerun cycle 211's setup (BLOCKS=[1,2,4,5,7,8], K=512, 4000 Adam steps on W_out/b_out) with eval-every-100-steps + best-state snapshot/restore across 3 seeds (42, 137, 2024). Settle the cycle 211 INCONCLUSIVE.

# Hypothesis
Cycle 211 achieved best PPL=59.680 at step 1500 (criterion ≤60 briefly met) but then overfit to 72.250 by step 4000. With proper per-eval-step early stopping and 3 seeds, the aggregated best should land near or slightly below 59.680 — tightening the verdict.

Item 226 criteria (VERBATIM from brief cycle_221.md):
- Confirmed if: best_ppl_mean ≤ 58.0.
- Refuted if: best_ppl_mean > 62.0.
- Inconclusive if: 58 < best_ppl_mean ≤ 62.

# Pre-flight Results
- P1 duplicate: NOT duplicate. Cycle 211 ran single seed, fixed budget, eval every 500, no best-state restoration. Cycle 221 formalizes early stopping with 3 seeds and eval cadence 100.
- P2 dependency: none.
- P3 ≤30 min: same compute envelope as cycle 211 (4000 steps × 6 blocks). Eval every 100 steps adds ~40 evals/seed × 3 seeds but each eval is cheap (100 seqs, 128 tokens). Cycle 211 completed within budget; cycle 221 fits.
- P4 program/prompts: no changes.

# Steps
1. Create `experiments/adam_6block_earlystop_cycle221.py` by copying structure of `experiments/adam_6block_cycle211.py`. Keep: tokenizer/dataset loader, `CompressedMLP`, `collect_mlp_io`, `fit_block_lstsq`, `eval_ppl`.
2. Configuration:
   - `BLOCKS = [1, 2, 4, 5, 7, 8]`
   - `K = 512`
   - `N_CALIB = 200`, `N_EVAL = 100`, `SEQ_LEN = 128`
   - `LR = 1e-4` (matches cycle 211)
   - `N_STEPS = 4000` (full budget, no truncation)
   - `EVAL_EVERY = 100`
   - `SEEDS = [42, 137, 2024]`
   - `OUTPUT_PATH = "autoresearch/results/cycle_221_adam_6block_earlystop.json"`
   - `SUMMARY_PATH = "autoresearch/results/cycle_221_summary.md"`
3. Wrap in `for seed in SEEDS:` loop. Per seed:
   a. `torch.manual_seed(seed)`, `random.seed(seed)`.
   b. Build fresh GPT-2 model, run `collect_mlp_io` on BLOCKS, replace each block's mlp via `fit_block_lstsq`.
   c. `init_ppl = eval_ppl(model, eval_ids)`.
   d. Freeze all params; unfreeze W_out/b_out for the 6 BLOCKS. Build Adam optimizer on these.
   e. Snapshot helper: `best_state = {b: (W_out.detach().clone().cpu(), b_out.detach().clone().cpu()) for b in BLOCKS}`; `best_ppl = init_ppl`; `best_step = 0`.
   f. Train 4000 steps, iterating `calib_shuffled` with `random.shuffle`. Every 100 steps call `eval_ppl`; append `{step, eval_ppl}` to `history`. If `eval_ppl < best_ppl`: update `best_ppl`, `best_step`, and refresh `best_state` (clone to CPU to avoid OOM).
   g. After step 4000: `final_ppl_at_step_4000 = eval_ppl(...)` (before restore).
   h. Restore best: for each b, copy cached tensors back into `model.transformer.h[b].mlp.W_out.data` and `.b_out.data` (move to DEVICE).
   i. `final_ppl_restored = eval_ppl(...)`; assert `abs(final_ppl_restored - best_ppl) < 1e-3` (sanity).
   j. Record per-seed dict: `{"seed": seed, "init_ppl": ..., "best_ppl": ..., "best_step": ..., "final_ppl_at_step_4000": ..., "history": [...]}`.
   k. `del model; torch.mps.empty_cache() if DEVICE == "mps"` between seeds.
4. Aggregate across seeds using `torch.tensor(...).mean()`/`.std()` for {init, best, best_step, final_at_4000}.
5. Verdict: use item 226 criteria above against `best_mean`.
6. Print comparison: cycle 211 single-seed best=59.680 vs cycle 221 mean ± std.
7. Write JSON to `autoresearch/results/cycle_221_adam_6block_earlystop.json` matching schema below.
8. Write `autoresearch/results/cycle_221_summary.md` with: verdict, per-seed table, aggregate row, trajectory description (per-seed step→PPL), comparison to cycle 211 (59.680) and 3-block joint baseline (46.556).

Stop after step 8; runner handles bookkeeping/manifest/commit.

## JSON schema
```
{
  "cycle": 221, "item": 226, "verdict": "...",
  "config": {"BLOCKS": [1,2,4,5,7,8], "K": 512, "N_CALIB": 200, "N_EVAL": 100,
             "SEQ_LEN": 128, "LR": 1e-4, "N_STEPS": 4000, "EVAL_EVERY": 100,
             "SEEDS": [42,137,2024]},
  "seeds": [
    {"seed": 42, "init_ppl": ..., "best_ppl": ..., "best_step": ...,
     "final_ppl_at_step_4000": ..., "history": [{"step":100,"eval_ppl":...}, ...]},
    ...
  ],
  "aggregate": {"init_mean": ..., "init_std": ..., "best_mean": ..., "best_std": ...,
                "best_step_mean": ..., "best_step_std": ...,
                "final_mean": ..., "final_std": ...},
  "cycle_211_baseline": 59.680
}
```

# Validation
- `init_ppl` per seed ≈ cycle 211's 6-block init (~86.7); all seeds should agree to <2 PPL (same calib, different RNG shouldn't affect init which is deterministic from topk+lstsq).
- `best_ppl` per seed should be ≤ 60 (cycle 211's implicit best at step 1500 was 59.680).
- With proper early stopping `best_ppl_mean` should be at or below 59.680, never worse.
- `final_ppl_at_step_4000` should reproduce cycle 211's ~72 overfitting on seed 42.
- After restore, `eval_ppl == best_ppl` (within 1e-3).
- Trainable params = 6 × (768×512 + 768) = 6 × 394 752 = 2 368 512 ≈ 2.37M (matches cycle 211's 2.36M).

# Artifacts
- `experiments/adam_6block_earlystop_cycle221.py`
- `autoresearch/results/cycle_221_adam_6block_earlystop.json`
- `autoresearch/results/cycle_221_summary.md`
- `autoresearch/results/cycle_221_manifest.txt` (runner writes this)

# Fallbacks
- Wall time >28 min: reduce `N_STEPS` to 3000 — cycle 211 best was at step 1500, so 3000 fully brackets it.
- Best-state snapshot OOM on MPS: already snapshotted to CPU (`.detach().clone().cpu()`); if still tight, store only the 6 W_out tensors (drop b_out diffs) — but 6 × 2.37M fp32 ≈ 57 MB on CPU, trivial.
- Eval every 100 too slow on MPS: fall back to every 200 (still 20 eval points, good enough to localize best step).
- If all seeds diverge before step 500: lower LR to 5e-5 and rerun (note in summary).

# Update bookkeeping
- Move item 226 from `autoresearch/agenda.md` NEXT/P0 to PAST/Cycle 221 with one-line outcome.
- Append TSV row to `autoresearch/results/results.tsv` (or equivalent ledger) with cycle, item, verdict, best_mean ± std, final_mean.
- Commit new script, JSON, summary, manifest, agenda, ledger, STATUS.md in one `[autoresearch] cycle 221: ...` commit.
- Refresh `autoresearch/STATUS.md` with verdict line.
