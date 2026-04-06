# Task

Item 228 (P1): Lower K with Adam correction — compression efficiency curve.
Test K=128 and K=256 with Adam W_out+b_out correction on BLOCKS=[1,4,7], using
the early-stop / best-state restore protocol from cycles 210/221. Compare to
the existing K=512 Adam baseline (cycle 210 best 46.556).

# Hypothesis

Quoted verbatim from agenda item 228 / brief cycle 223:

- Confirmed if K=256 + Adam achieves PPL ≤ 52.0 (+3.5% vs float 53.895).
- Refuted if K=256 + Adam PPL > 58.0.
- Inconclusive if 52 < K=256 best_mean ≤ 58.

The verdict criterion is K=256-specific. K=128 is run for the efficiency curve
only; it has no pass/fail criterion.

# Pre-flight Results

- P1 (duplicate): K=512+Adam exists (cycles 206/210/221). K=128 and K=256 with
  Adam are new. Cycle 199 K-sweep used no Adam. Cycle 205 K-sweep was block 1
  alone, not 3-block. NOT a duplicate.
- P2 (dependency): none.
- P3 (≤30 min wall time): 2 K values × 3 seeds × 4000 steps Adam, comparable
  to cycle 221 (which ran 6-block × 3 seeds × 4000 steps). Tight; mitigate
  via fallback ordering (run K=256 first).
- P4 (program/prompts): no changes needed.

# Steps

1. Create `experiments/adam_lower_k_cycle223.py`. Reuse loader, `CompressedMLP`,
   `collect_mlp_io`, `fit_block_lstsq`, `eval_ppl` and per-seed `run_seed`
   structure from `experiments/adam_6block_earlystop_cycle221.py` verbatim.
2. Constants: `BLOCKS=[1,4,7]`, `K_VALUES=[256, 128]` (K=256 first per fallback),
   `N_CALIB=200`, `N_EVAL=100`, `SEQ_LEN=128`, `LR=1e-4`, `N_STEPS=4000`,
   `EVAL_EVERY=100`, `SEEDS=[42, 137, 2024]`.
3. Float baseline once. Outer loop over K, inner loop over seed:
   - Build BLOCKS=[1,4,7] topk+lstsq compressed model with this K.
   - Eval init PPL.
   - Freeze all params, unfreeze W_out + b_out of the 3 compressed MLPs.
   - Adam (LR=1e-4) for 4000 steps; eval every 100; snapshot best state on CPU.
   - At end: restore best state, assert restored PPL == best_ppl (±1e-3).
   - Record: init_ppl, best_ppl, best_step, final_ppl_at_4000, history[].
4. Aggregate per K: init_mean/std, best_mean/std, best_step_mean/std, final_mean/std.
5. Verdict (K=256 only):
   - CONFIRMED if K256 best_mean ≤ 52.0
   - REFUTED if K256 best_mean > 58.0
   - INCONCLUSIVE otherwise
6. Compression efficiency curve: for each K in {128, 256, 512}, emit
   (K, trainable_params_total, best_ppl_mean, pct_vs_float). K=512 row uses
   the cycle 210 number (46.556) as a hard-coded baseline; K=128 and K=256 from
   this run.
7. Write JSON to `autoresearch/results/cycle_223_lower_k_adam.json` (schema below).
8. Write summary markdown to `autoresearch/results/cycle_223_summary.md` with the
   verdict, the efficiency curve table, per-seed details, and a comparison to
   the cycle 206/210 K=512 numbers.

# JSON schema

```
{
  "cycle": 223, "item": 228, "verdict": "CONFIRMED|REFUTED|INCONCLUSIVE",
  "config": {"BLOCKS": [1,4,7], "K_VALUES": [256,128], "LR": 1e-4,
             "N_STEPS": 4000, "EVAL_EVERY": 100, "SEEDS": [42,137,2024],
             "N_CALIB": 200, "N_EVAL": 100, "SEQ_LEN": 128},
  "float_ppl": <float>,
  "k_512_baseline": {"best_mean": 46.556, "source": "cycle 210"},
  "results": [
    {"K": 256, "trainable_params_per_block": 197376, "total_trainable": 592128,
     "seeds": [{"seed": 42, "init_ppl": ..., "best_ppl": ...,
                "best_step": ..., "final_ppl_at_4000": ..., "history": [...]}, ...],
     "aggregate": {"init_mean": ..., "init_std": ...,
                   "best_mean": ..., "best_std": ...,
                   "best_step_mean": ..., "best_step_std": ...,
                   "final_mean": ..., "final_std": ...}},
    {"K": 128, "trainable_params_per_block": 98688, "total_trainable": 296064, ...}
  ]
}
```

# Validation

- Init PPL at K=128 should be much higher than K=512 init (61.746 from cycle 206).
- Init PPL at K=256 should sit between K=128 init and K=512 init.
- Trainable count K=256: 256×768 + 768 = 197376 per block × 3 = 592128.
- Trainable count K=128:  98688 per block × 3 = 296064.
- Best-state restore must round-trip (assert as in cycle 221).

# Artifacts

- experiments/adam_lower_k_cycle223.py
- autoresearch/results/cycle_223_lower_k_adam.json
- autoresearch/results/cycle_223_summary.md
- autoresearch/results/cycle_223_manifest.txt (written by execute step)

# Fallbacks

- If wall time tight, K=256 runs first (verdict-critical). K=128 second.
- If only K=256 completes: report partial efficiency curve with the verdict
  for K=256 and K=128 marked "not run".
- Verdict mapping is unconditional: K256 best_mean ≤ 52 → CONFIRMED;
  > 58 → REFUTED; otherwise INCONCLUSIVE.

# Update bookkeeping

- Move item 228 from NEXT/P1 to PAST/Cycle 223 in `autoresearch/agenda.md`.
- Append TSV row to `autoresearch/results.tsv` (cycle, item, verdict, key metric).
- Commit cycle 223 plan + experiment + results + manifest with standard message.
- Update `autoresearch/STATUS.md` with cycle 223 line.
