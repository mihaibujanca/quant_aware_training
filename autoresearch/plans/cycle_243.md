# Task

Cycle 243 / item 253: per-block PPL attribution under the cycle 221 best-state 6-block joint Adam fit (BLOCKS=[1,2,4,5,7,8], K=512). Measure how much each compressed block contributes to the 6.0 PPL gap (53.895 → 59.515) by restoring its float MLP one block at a time.

# Hypothesis

> Probe the 6-block 59.5 ceiling — per-block PPL decomposition under the [1,2,4,5,7,8] joint fit. Analysis item, no new training. Load cycle 221's best-state joint Adam snapshot for BLOCKS=[1,2,4,5,7,8] K=512 W_out+b_out (should be in `autoresearch/results/cycle_221_*.pt` or reconstructable from the logged seed 42 run). Starting from the fully compressed cascade at 59.515 PPL, measure the eval PPL contribution of each compressed block by UN-compressing one block at a time (restore the GPT-2 FFN for that block only, keep the other 5 compressed) on the cycle 221 eval set (`all_ids[200:300]`, SEQ_LEN=128, seed 42). Report six numbers: `ppl_if_block_X_restored` for X ∈ {1,2,4,5,7,8}, plus the full-float baseline (53.895) and the fully-compressed baseline (59.515). The largest ΔPPL drops identify the "bottleneck blocks" in the cascade — the blocks whose compression contributes the most to the 13-PPL 3→6 gap. Also record: (a) per-block mean residual MSE at the cycle 221 best state, (b) the `sorted(-ΔPPL)` curve to see if one or two blocks dominate vs uniform contribution. If cycle 221's snapshot isn't durable on disk, re-fit joint Adam with seed 42 only (cheaper than 3-seed; ≤10 min) to regenerate it.

# Pre-flight Results

- `cycle_221_adam_6block_earlystop.json` contains trajectory + best_step (1300 for seed 42, best_ppl=59.391) but **no W_out/b_out tensors** — no `.pt` snapshot exists. Refit at seed 42 only is required.
- Cycle 221 reference: float_ppl=53.895, fully_compressed best_ppl seed 42=59.391 (mean 59.515).
- Eval set: `all_ids[200:300]`, SEQ_LEN=128, calib `all_ids[:200]`. Reuse loader verbatim.
- Refit budget: 4000 steps × seed 42 only ≈ 10 min on MPS.

# Steps

1. Create `experiments/6block_per_block_attribution_cycle243.py`. Imports + data loader + `CompressedMLP` + `collect_mlp_io` + `fit_block_lstsq` + `eval_ppl` copied verbatim from `experiments/adam_6block_earlystop_cycle221.py`. BLOCKS=[1,2,4,5,7,8], K=512, SEQ_LEN=128, N_CALIB=200, N_EVAL=100, LR=1e-4, N_STEPS=4000, EVAL_EVERY=100, SEED=42.
2. **Phase 0 — snapshot acquisition.** Always refit (no `.pt` on disk):
   - Build float GPT-2, eval `float_baseline_ppl` (assert ≈ 53.895).
   - Hook MLP I/O on calib (200 seqs).
   - Replace MLPs at BLOCKS with `fit_block_lstsq` outputs.
   - Freeze all params, enable grad on `W_out`, `b_out` for BLOCKS only. Adam(lr=1e-4). 4000 steps over shuffled calib_ids with EVAL_EVERY=100. Snapshot `(W_out, b_out)` per block on each new best.
   - Restore best state. `assert |restored_ppl - best_ppl| < 1e-3`. Record `fully_compressed_ppl = restored_ppl`.
3. **Phase 1 — keep references.** Build a fresh float GPT-2 (`model_float`). For each block X in BLOCKS, store `float_mlp_X = copy.deepcopy(model_float.transformer.h[X].mlp)` (so the compressed model can restore them individually).
4. **Phase 2 — per-block restoration scan.** For each X ∈ [1,2,4,5,7,8]:
   - `saved = compressed_model.transformer.h[X].mlp`
   - `compressed_model.transformer.h[X].mlp = float_mlp_X` (in place, on DEVICE)
   - `ppl_if_X = eval_ppl(compressed_model, eval_ids)`
   - Restore: `compressed_model.transformer.h[X].mlp = saved`
   - Sanity: re-eval after restore matches `fully_compressed_ppl` within 1e-4.
   - Record `delta_X = fully_compressed_ppl - ppl_if_X`.
5. **Phase 3 — per-block residual MSE.** Re-run `collect_mlp_io` with `model_float` (float MLP outputs) on calib data to get `proj_outs_float[X]`. Re-run on the cycle-221 best-state compressed model to get `proj_outs_comp[X]`. For each X compute `mse_X = ((proj_outs_float[X] - proj_outs_comp[X]) ** 2).mean().item()`. Report 6 numbers.
6. **Verdict.** any X with `ppl_if_X ≤ 56.0` → CONFIRMED. all 6 deltas in [0.5, 1.5] → REFUTED. else INCONCLUSIVE.
7. Write JSON + summary markdown with the 6-row table sorted by `delta_X` descending: columns `block | ppl_if_X_restored | delta_ppl | mse | rank`.

# Validation

- `fully_compressed_ppl` matches cycle 221 seed-42 best (59.391, allow ±0.05 since optimisation has some FP nondeterminism on MPS).
- `float_baseline_ppl` ≈ 53.895 (±0.001).
- Each `ppl_if_X_restored` lies in [53.895, 59.515] (monotone — restoring a float block never makes things worse).
- Sum of deltas may exceed (or fall short of) the 6.0 PPL gap — non-additivity is expected and noted, not a failure.
- Restore-after-swap roundtrip eval matches `fully_compressed_ppl` to 1e-4 (asserts no aliasing bug).

# Artifacts

- `experiments/6block_per_block_attribution_cycle243.py`
- `autoresearch/results/cycle_243_per_block_attribution.json`
- `autoresearch/results/cycle_243_summary.md`
- `autoresearch/results/cycle_243_manifest.txt` (written by execute step)

# Fallbacks

- If the seed-42 refit lands at best_ppl > 59.7 (>0.3 PPL above cycle 221 seed-42 reference), still proceed — record actual `fully_compressed_ppl` and use it as the baseline for delta computation. Note divergence in summary.
- If MPS OOM during double-loaded `model_float` + `compressed_model`, move `model_float` to CPU and only the active swapped MLP to MPS for the duration of one eval.
- If refit > 15 min wall-clock, halve N_STEPS to 2000 (best step in cycle 221 was ~1300, plenty of headroom) and rerun.

# Update bookkeeping

- Move item 253 from NEXT/P2 to PAST/Cycle 243 in `autoresearch/agenda.md`.
- Add TSV row to `autoresearch/results/results.tsv`.
- Commit results + script.
- Update `autoresearch/STATUS.md` with the verdict + ranking of bottleneck blocks.
