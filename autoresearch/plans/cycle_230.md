# Task

Cycle 230 — Item 240 (P0): K ∈ {8, 4, 2, 1} Adam compression cliff sweep — find the absolute floor. Extend the cycle 225 efficiency curve (K=64/32/16) into the sub-16 regime. New script `experiments/adam_k_floor_cycle230.py` copied verbatim from `experiments/adam_lower_k_cliff_cycle225.py` (only `K_VALUES`, verdict, and bookkeeping change).

# Hypothesis

> 240. **K ∈ {8, 4, 2, 1} Adam compression cliff sweep — find the absolute floor**
>    - Extend the cycle 225 curve (K=512→16 monotone, K=16 first to cross +2.3% above float) into the sub-16 regime. BLOCKS=[1,4,7], K ∈ {8, 4, 2, 1} Adam W_out+b_out, LR=1e-4, 4000 steps, best-state restore, 3 seeds (42, 137, 2024). Reuse `experiments/adam_lower_k_cliff_cycle225.py` as template; same calib/eval split as cycles 206/223/225. Report init_ppl and best_ppl per K with mean ± std, and extend the {512,256,128,64,32,16,...} table. Identify the K where the surface stops learning at all (init = best within noise) vs. where it catastrophically breaks. Params per block at K=1: 1×768 + 768 = 1,536 trainable for all three blocks combined — a stress test of Adam's ability to absorb aggressive compression.
>    - Type: empirical
>    - Confirmed if: Adam W_out correction at K=4 stays within +10% of float (≤59.28 PPL). Pinpoints the compression floor as K≤4.
>    - Refuted if: K=8 already exceeds +15% of float (≥61.98 PPL) — cliff is steeper than the 1-2 PPL/halving trend predicts.
>    - Inconclusive if: K=8 lands between +2.3% (the K=16 value) and +15% — curve continues flattening into sub-16 territory but does not localize a single cliff.

# Pre-flight Results

P1-P4 clean. K_VALUES ∈ {8,4,2,1} fully disjoint from cycles 210 (K=512), 223 (K=128/256), 225 (K=16/32/64). Template `experiments/adam_lower_k_cliff_cycle225.py` exists and is the direct base. Same calib/eval split (`all_ids[:200]` / `all_ids[200:300]`, SEQ_LEN=128).

# Steps

1. Create `experiments/adam_k_floor_cycle230.py` by copying `experiments/adam_lower_k_cliff_cycle225.py` verbatim, then:
   - Set `K_VALUES = [8, 4, 2, 1]`.
   - Set `OUTPUT_PATH = "autoresearch/results/cycle_230_k_floor.json"`.
   - Set `SUMMARY_PATH = "autoresearch/results/cycle_230_summary.md"`.
   - Update header docstring to say cycle 230 / item 240 and the new verdict thresholds (CONFIRMED: K=4 best_mean ≤ 59.28; REFUTED: K=8 best_mean ≥ 61.98; else INCONCLUSIVE).
   - Set `result["cycle"] = 230`, `result["item"] = 240`.
   - Extend `REFERENCE_CURVE` to include K=16 (55.133, cycle 225), K=32 (53.038, cycle 225), K=64 (50.781, cycle 225), K=128 (48.989, cycle 223), K=256 (47.509, cycle 223), K=512 (46.556, cycle 210).
2. Keep all other hyperparameters identical: `BLOCKS=[1,4,7]`, `N_CALIB=200`, `N_EVAL=100`, `SEQ_LEN=128`, `LR=1e-4`, `N_STEPS=4000`, `EVAL_EVERY=100`, `SEEDS=[42,137,2024]`. Same calib/eval split (`all_ids[:200]` / `all_ids[200:300]`).
3. Reuse `CompressedMLP`, `collect_mlp_io`, `fit_block_lstsq`, `eval_ppl`, `run_seed` unchanged. Per-seed loop: build topk+lstsq init → eval init → freeze base, train W_out+b_out for 4000 Adam steps with eval every 100 → snapshot best → restore best → assert restore round-trip. Record init_ppl, best_ppl, best_step, final_ppl_at_4000, history.
4. Aggregate per K (init/best/best_step/final mean±std across 3 seeds).
5. Rewrite verdict block per item 240 criteria:
   - `k8 = best_means[8]`, `k4 = best_means[4]`.
   - CONFIRMED if `k4 ≤ 59.28`.
   - REFUTED if `k8 ≥ 61.98`.
   - INCONCLUSIVE otherwise (K=8 in (55.133, 61.98) or K=4 > 59.28 but K=8 < 61.98).
6. Write JSON to `autoresearch/results/cycle_230_k_floor.json` with config, float baseline, all K results (per-seed + aggregate), and the full `reference_curve` block covering K ∈ {16, 32, 64, 128, 256, 512}.
7. Write summary markdown to `autoresearch/results/cycle_230_summary.md` with the full extended efficiency-curve table over K ∈ {512, 256, 128, 64, 32, 16, 8, 4, 2, 1} (sorted high→low), per-K aggregates, and a "Floor diagnostic" section flagging any K where `|best_mean - init_mean| < 1.0` (Adam failed to learn) vs. `best_mean > 2 × float_ppl` (catastrophic break).

# Validation

- `init_ppl` strictly grows as K shrinks (1 > 2 > 4 > 8 > 16 > 32 > ... > 512). Note in summary if violated.
- `trainable_params_per_block == K*768 + 768`: K=8→6,912; K=4→3,840; K=2→2,304; K=1→1,536. `total_trainable == 3 * per_block`.
- Best-state restore round-trip: `assert abs(final_ppl_restored - best_ppl) < 1e-3` (template already enforces).
- Float PPL ≈ 53.895 (matches cycles 210/223/225).
- Verdict label matches printed K=8 and K=4 best_mean values.

# Artifacts

- `experiments/adam_k_floor_cycle230.py`
- `autoresearch/results/cycle_230_k_floor.json`
- `autoresearch/results/cycle_230_summary.md`
- `autoresearch/results/cycle_230_manifest.txt` (written by execute step)

# Fallbacks

- If K=1 takes too long (unlikely — fewer params means faster step): drop K=1, keep K∈{8,4,2}. Verdict (uses K=8, K=4) still well-defined.
- If wall time tight: also drop K=2, keep K∈{8,4}. Minimum viable set to evaluate verdict.
- If MPS OOM (very unlikely at small K): fall back to CPU. Template's `del`/`empty_cache` already per-seed.

# Update bookkeeping

- Move item 240 from `agenda.md` NEXT/P0 → PAST/Cycle 230 with verdict (and DELETE from NEXT to avoid the runner-copy bug).
- Append TSV row to `autoresearch/program.md` cycle log.
- Update `autoresearch/STATUS.md` and write `autoresearch/results/cycle_230_manifest.txt` with all artifact hashes.
- Single commit per cycle convention: `[autoresearch] cycle 230: K=8/4/2/1 Adam floor — <VERDICT>`.
