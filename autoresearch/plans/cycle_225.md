# Task

Cycle 225 — Item 235 (P0): K=64/32/16 Adam compression cliff sweep. Extend the cycle 223 efficiency curve (K=512/256/128) into the small-K regime to find where Adam W_out+b_out correction finally fails to beat float. New script `experiments/adam_lower_k_cliff_cycle225.py` adapted from `experiments/adam_lower_k_cycle223.py` (only `K_VALUES` and bookkeeping change).

# Hypothesis

> 235. **K=64/32/16 Adam compression cliff sweep**
>    - Cycle 223 confirmed K=128 still beats float with Adam W_out correction (best_mean 48.989 vs float 53.895). The compression efficiency curve was monotone but very flat between K=512 → K=128. Push the curve until it breaks. BLOCKS=[1,4,7], K ∈ {64, 32, 16}, Adam W_out+b_out, 3 seeds (42/137/2024), 4000 steps, eval every 100, best-state restore. Same calib/eval split as cycle 223 (`all_ids[:200]` / `all_ids[200:300]`, SEQ_LEN=128).
>    - Type: empirical
>    - Confirmed if: at least one K ≤ 64 still beats float (best_mean ≤ 53.0) — channel capacity is not the bottleneck even at 8× lower than K=512
>    - Refuted if: all three K give best_mean ≥ 60.0 — there is a real channel-capacity floor below K=128
>    - Inconclusive if: best_mean lies in 53-60 PPL for the smallest viable K

# Pre-flight Results

P1-P4 clean. Cycle 223 ran K∈{128,256}; this cycle runs K∈{16,32,64} — fully disjoint, no overlap with prior cycle results. Template script `experiments/adam_lower_k_cycle223.py` exists and is the direct base.

# Steps

1. Create `experiments/adam_lower_k_cliff_cycle225.py` by copying `experiments/adam_lower_k_cycle223.py` verbatim, then:
   - Set `K_VALUES = [64, 32, 16]` (K=64 first to resolve the cliff verdict early).
   - Set `OUTPUT_PATH = "autoresearch/results/cycle_225_lower_k_cliff.json"`.
   - Set `SUMMARY_PATH = "autoresearch/results/cycle_225_summary.md"`.
   - Update header docstring/comments to say cycle 225 / item 235.
   - Update `result["cycle"] = 225`, `result["item"] = 235`.
2. Keep all other hyperparameters identical: `BLOCKS=[1,4,7]`, `N_CALIB=200`, `N_EVAL=100`, `SEQ_LEN=128`, `LR=1e-4`, `N_STEPS=4000`, `EVAL_EVERY=100`, `SEEDS=[42,137,2024]`. Same calib/eval split (`all_ids[:200]` / `all_ids[200:300]`).
3. Reuse `CompressedMLP`, `collect_mlp_io`, `fit_block_lstsq`, `eval_ppl`, `run_seed` unchanged. Per-seed loop = topk+lstsq init → eval init → freeze base, train W_out+b_out for 4000 Adam steps with eval every 100 → snapshot best → restore best → assert restore round-trip.
4. Aggregate per K (init/best/best_step/final mean±std across 3 seeds).
5. Verdict logic per item 235:
   - `min_best = min(K results' best_mean for K in {64,32,16})`
   - CONFIRMED if `min_best ≤ 53.0`
   - REFUTED if `all best_mean ≥ 60.0`
   - INCONCLUSIVE otherwise (smallest viable K's best_mean in (53, 60))
6. Write JSON to `autoresearch/results/cycle_225_lower_k_cliff.json` containing config, float baseline, all K results (per-seed + aggregate), and a `reference_curve` block with cycle 223/210 numbers: K=128 → 48.989, K=256 → 47.509, K=512 → 46.556 (sources annotated).
7. Write summary markdown to `autoresearch/results/cycle_225_summary.md` with the extended efficiency-curve table over K ∈ {512, 256, 128, 64, 32, 16} (sorted by K), per-K aggregates, and a "Comparison to cycle 223" section listing the deltas.

# Validation

- `init_ppl` strictly grows as K shrinks (16 > 32 > 64 > 128 > 256 > 512). Note in summary if violated.
- `trainable_params_per_block == K*768 + 768` for each K (49,920 / 25,344 / 13,056 for K=64/32/16). `total_trainable == 3 * per_block`.
- Best-state restore round-trip: `assert abs(final_ppl_restored - best_ppl) < 1e-3` (template already enforces this).
- Float PPL ≈ 53.895 (matches cycles 210/223).
- Verdict label matches the K=64/32/16 best_mean values printed in the verdict block.

# Artifacts

- `experiments/adam_lower_k_cliff_cycle225.py`
- `autoresearch/results/cycle_225_lower_k_cliff.json`
- `autoresearch/results/cycle_225_summary.md`
- `autoresearch/results/cycle_225_manifest.txt` (written by execute step)

# Fallbacks

- If wall time runs tight (template runs ≈3 K × 3 seeds × 4000 steps): drop K=16 first (least informative for cliff location), keep K=64 and K=32. Verdict is still well-defined on the surviving K subset.
- If K=64 already gives `best_mean ≤ 53.0`: verdict is CONFIRMED regardless of K=32/16, but still run K=32 and K=16 for the curve (do not short-circuit the loop).
- If MPS OOM at small K (unlikely — smaller W_out): fall back to CPU. Template's `del`/`empty_cache` already in place per seed.

# Update bookkeeping

- Move item 235 from `agenda.md` NEXT/P0 → PAST/Cycle 225 with verdict.
- Append TSV row to `autoresearch/program.md` cycle log.
- Update `autoresearch/STATUS.md` and write `autoresearch/results/cycle_225_manifest.txt` with all artifact hashes.
- Single commit per cycle convention: `[autoresearch] cycle 225: K=64/32/16 Adam cliff — <VERDICT>`.
