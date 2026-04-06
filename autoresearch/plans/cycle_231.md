# Task

Run K=128 joint Adam correction at 6 blocks to test whether the flat ceiling seen at K=512 (cycle 221) persists at 4× lower compression budget. Mirror cycle 221's protocol exactly — only K changes.

# Hypothesis

> 241. **K=128 Adam correction at 6 blocks — does the flat curve hold at higher block count?**
>    Cycle 223 (K=128) beat float at 3 blocks (48.989 vs 53.895); cycle 221 showed joint Adam at 6 blocks K=512 hits a 59.5 PPL ceiling. Test whether the K=128 compression survives the 3→6 block cascade jump or whether the lower-K surface collapses at higher cascade depth. BLOCKS=[1,2,4,5,7,8] K=128 Adam W_out+b_out, LR=1e-4, 4000 steps with best-state restore and eval-every-100, 3 seeds. Reuse `experiments/adam_6block_earlystop_cycle221.py` as template but swap K. Report init, best, best_step, final, and compare against cycle 221's K=512 ceiling (59.515 ± 0.094). Success would give a 4× more compressed alternative to K=512 at matched PPL.

# Pre-flight Results

Clean. P1–P4 green. Cycle 221 K=512 at 6 blocks → 59.515 ± 0.094 (INCONCLUSIVE vs its own ≤58.0 bar but a valid ceiling to compare against). Cycle 223 K=128 at 3 blocks → 48.989 < float 53.895. Template script exists and runs under 2h on MPS.

# Steps

1. Copy `experiments/adam_6block_earlystop_cycle221.py` → `experiments/adam_6block_k128_cycle231.py`. Change `K = 512` → `K = 128`. Update `OUTPUT_PATH` to `autoresearch/results/cycle_231_6block_k128.json` and `SUMMARY_PATH` to `autoresearch/results/cycle_231_summary.md`. Update header docstring to reference cycle 231 / item 241 and the new verdict thresholds (≤61.0 / ≥65.0 / inconclusive). Update `cycle` and `item` fields in the result JSON to 231 and 241. Leave everything else identical.
2. Config constants (verify unchanged): `BLOCKS=[1,2,4,5,7,8]`, `K=128`, `N_CALIB=200`, `N_EVAL=100`, `SEQ_LEN=128`, `LR=1e-4`, `N_STEPS=4000`, `EVAL_EVERY=100`, `SEEDS=[42,137,2024]`.
3. Per seed (existing `run_seed`): topk+lstsq init (`fit_block_lstsq` with K=128), eval init PPL, freeze all + unfreeze `W_out`/`b_out` on BLOCKS, Adam 4000 steps with eval every 100, snapshot best state on improvement, restore best at end. Record init, best, best_step, final@4000, history list.
4. Aggregate across 3 seeds (existing `agg`): best_mean ± best_std. Apply item 241 verdict: Confirmed if best_mean ≤ 61.0, Refuted if ≥ 65.0, else Inconclusive. Also report delta vs 59.515 (cycle 221 K=512 ceiling).
5. Write JSON (`cycle_231_6block_k128.json`) and summary markdown (`cycle_231_summary.md`) using the existing write paths, adjusted.

# Validation

- Init PPL expected in ~88–90 range (lower K → worse lstsq init than cycle 221's ~86.7 at K=512).
- Trainable params per block at K=128: `W_out` 768×128 + `b_out` 768 = 99,072. Total across 6 blocks = 594,432 (~0.59M), roughly half of cycle 221's 2.36M.
- Restore-best assertion `abs(final_ppl_restored - best_ppl) < 1e-3` must hold for all 3 seeds.
- Verdict thresholds: ≤61.0 Confirmed, ≥65.0 Refuted, else Inconclusive (matches brief and item 241).

# Artifacts

- `experiments/adam_6block_k128_cycle231.py`
- `autoresearch/results/cycle_231_6block_k128.json`
- `autoresearch/results/cycle_231_summary.md`
- `autoresearch/results/cycle_231_manifest.txt` (written by execute stage)

# Fallbacks

- Wall time: cycle 221 fit under the loop budget with the same shape (6 blocks × 4000 steps × 3 seeds). K=128 should be strictly faster (smaller matmul in CompressedMLP forward). No reduction planned.
- If init PPL > 120 on any seed (lstsq bug), abort that seed and rerun; do not continue into Adam.

# Update bookkeeping

- Move item 241 from NEXT/P0 to PAST/Cycle 231 with verdict, commit hash, and artifact list.
- Append TSV row to `autoresearch/results/cycles.tsv`.
- Update `autoresearch/STATUS.md` with cycle 231 one-liner.
- Commit: `[autoresearch] cycle 231: K=128 Adam 6-block — <verdict>`.
