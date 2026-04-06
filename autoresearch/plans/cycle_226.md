# Task

Cycle 226, item 236 (P0). KD distillation baseline rerun with corrected loss normalization. Same trainable surface as cycle 224 (BLOCKS=[1,4,7], K=512, FullCompressedMLP — W_out+b_out+W_fc+b_fc). Fix the `F.kl_div(..., reduction='batchmean')` bug by reshaping student/teacher logits to `(B*T, V)` so KD divides by per-token count and matches CE scale. Run α-sweep at 2000 steps × 1 seed, then 4000 steps × 3 seeds at the best α.

# Hypothesis

Item 236 (verbatim): "KD distillation baseline rerun with corrected loss normalization. Cycle 224's distillation experiment was downgraded to INCONCLUSIVE because `F.kl_div(..., reduction='batchmean')` on `(B=1, T=128, V=50257)` divides only by B=1, returning the SUM over T·V ≈ 6.4M entries (~30-50) while CE averages over tokens (~4). The 50/50 mix was ~10× KD-weighted in practice. Rerun with `logits.reshape(B*T, V)` before `kl_div` (or `reduction='mean'`) so KD and CE are on the same per-token scale. Same trainable surface as cycle 224 (BLOCKS=[1,4,7] K=512, W_out+b_out+W_fc+b_fc, ≈2.36M params), tau=2.0, LR=1e-4. Run a short KD-weight sweep (α ∈ {0.0, 0.5, 1.0}) at 2000 steps × 1 seed to verify the loss is wired correctly, then 4000 steps × 3 seeds at the best α."

# Pre-flight Results

- P1 duplicate: cycle 224 was downgraded to INCONCLUSIVE due to method failure (broken KD normalization). This is the corrected rerun. NOT duplicate.
- P2 dependency: none.
- P3 ≤30 min: 2000 steps × 3 (α sweep, single seed) + 4000 steps × 3 seeds (full run) = 18k step-equivalents (with dual forward teacher+student for α>0). Cycle 224 fit 30k step-equivalents in budget, so feasible.
- P4 program/prompts: no changes.

# Steps

1. Create `experiments/distillation_corrected_cycle226.py` based on `experiments/distillation_baseline_cycle224.py`. Keep all data loading, `FullCompressedMLP`, `collect_mlp_io`, `fit_block_lstsq`, `eval_ppl`, `snapshot_compressed`, `restore_compressed`, BLOCKS=[1,4,7], K=512, N_CALIB=200, N_EVAL=100, SEQ_LEN=128, LR=1e-4, TAU=2.0 unchanged.
2. KD loss (CORRECTED — replace lines 219–229 of cycle 224 inside `run_seed`'s training step):
   ```python
   B, T, V = student_logits.shape
   student_log_softmax = F.log_softmax(student_logits.reshape(B * T, V) / TAU, dim=-1)
   teacher_softmax = F.softmax(teacher_logits.reshape(B * T, V) / TAU, dim=-1)
   kd_loss = F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * (TAU ** 2)
   ce_loss = F.cross_entropy(
       student_logits[:, :-1].reshape(-1, V),
       ids[:, 1:].reshape(-1),
   )
   total_loss = (1.0 - alpha) * ce_loss + alpha * kd_loss
   ```
   Note: with reshape `(B*T, V)`, batchmean divides by `B*T`, giving per-token KD loss comparable to CE.
3. α-sweep phase: refactor `run_seed` to take `alpha` and `n_steps`/`eval_every` args. Run with SEED=42, n_steps=2000, eval_every=200, α ∈ {0.0, 0.5, 1.0}. For each α, train fresh (re-init student via fresh `GPT2LMHeadModel.from_pretrained` + `fit_block_lstsq`), record best PPL and best step. Pick α* = argmin best_ppl across the sweep.
4. Full run phase: SEEDS=[42, 137, 2024], α=α* from sweep, n_steps=4000, eval_every=100, best-state snapshot/restore (already in cycle 224 helper functions).
5. Aggregate full-run init/best/best_step mean ± std (as in cycle 224 lines 277–284).
6. Verdict per item 236: CONFIRMED if best_mean ≤ 44.0; REFUTED if best_mean ≥ 47.0; else INCONCLUSIVE.
7. Write JSON to `autoresearch/results/cycle_226_distillation_corrected.json` with schema below.
8. Write summary markdown to `autoresearch/results/cycle_226_summary.md` with α-sweep table, full-run per-seed table, comparison vs cycle 213 (46.424) and cycle 224 broken (60.938).

JSON schema:
```
{
  "cycle": 226, "item": 236, "verdict": "CONFIRMED|REFUTED|INCONCLUSIVE",
  "config": {"BLOCKS":[1,4,7],"K":512,"LR":1e-4,"TAU":2.0,
             "SWEEP_STEPS":2000,"SWEEP_EVAL_EVERY":200,"SWEEP_ALPHAS":[0.0,0.5,1.0],
             "FULL_STEPS":4000,"FULL_EVAL_EVERY":100,"SEEDS":[42,137,2024],
             "N_CALIB":200,"N_EVAL":100,"SEQ_LEN":128},
  "alpha_sweep": [{"alpha":0.0,"best_ppl":...,"best_step":...}, ...],
  "best_alpha": ...,
  "full_run": {
    "seeds": [{"seed":42,"init_ppl":...,"best_ppl":...,"best_step":...,"history":[...]}, ...],
    "aggregate": {"init_mean":...,"best_mean":...,"best_std":...,"best_step_mean":...}
  },
  "baselines": {"cycle_213_pure_ce": 46.424, "cycle_224_broken_kd": 60.938, "float": ...}
}
```

# Validation

- α=0 sweep cell should reproduce cycle 213's 46.424 ± noise (sanity check that the surface and CE path are intact).
- KD loss magnitude after the reshape fix should be O(1–10), not O(30–50) (printed first 5 steps as in cycle 224 lines 235–237).
- Init PPL ~61.7 (matches cycle 224 init).
- Restore-best assertion `abs(best_ppl_restored - best_ppl) < 1e-3` per seed.

# Artifacts

- `experiments/distillation_corrected_cycle226.py`
- `autoresearch/results/cycle_226_distillation_corrected.json`
- `autoresearch/results/cycle_226_summary.md`
- `autoresearch/results/cycle_226_manifest.txt` (written by execute step)

# Fallbacks

- If α-sweep takes too long: reduce sweep to α ∈ {0.5, 1.0} (skip 0.0 since it's just cycle 213 reproduction).
- If wall time over budget for full run: reduce to 2 seeds [42, 137] or 3000 steps with eval_every=100.

# Update bookkeeping

- Move item 236 from `agenda.md` NEXT/P0 to PAST/Cycle 226 with the verdict line.
- Append TSV row to `autoresearch/results/all_cycles.tsv` (cycle 226, item 236, verdict, best_mean, delta vs c213).
- Commit with `[autoresearch] cycle 226: KD corrected normalization — <verdict>`.
- Update `STATUS.md` cycle counter.
