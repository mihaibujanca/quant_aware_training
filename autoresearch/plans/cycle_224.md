# Task

Item 229 (P2): KD distillation baseline for 3-block compression. Train W_out + W_fc (matched
to cycle 213) on a mixed KD+CE loss for 10000 steps × 3 seeds, BLOCKS=[1,4,7] K=512,
teacher=float GPT-2, student=topk+lstsq init. Compare against the cycle 206/210/213 CE-loss
ceiling.

# Hypothesis

Soft-target KD supervision (KL on softened logits from float teacher) gives a meaningfully
better optimum than hard-token CE for the W_out+W_fc trainable surface.

Item 229 thresholds (verbatim):
- Confirmed if: KD achieves PPL ≤44.0 (meaningfully better than Adam W_out correction 46.556)
- Refuted if: KD PPL ≥47.0 (Adam W_out correction is already at the ceiling)
- Inconclusive if: PPL 44-47

# Pre-flight Results

- P1 duplicate: cycle 213 trained W_out+W_fc with CE loss → 46.424. KD loss is NEW. Not a duplicate.
- P2 dependency: none.
- P3 ≤30 min: 10000 steps × 3 seeds × dual-forward (teacher + student) is TIGHT on MPS. Fallback
  to 5000 steps allowed (cycle 213 W_out+W_fc best was at step 1500, so 5000 is still ample).
- P4 program/prompts: no changes.

# Steps

1. Create `experiments/distillation_baseline_cycle224.py`. Reuse cycle 206/213 init helpers
   (`collect_mlp_io`, `fit_block_lstsq`, `eval_ppl`, `FullCompressedMLP` with W_fc+W_out
   trainable). BLOCKS=[1,4,7], K=512, N_CALIB=200, N_EVAL=100, SEQ_LEN=128.
2. Build the calib/eval split once (deterministic from tokenizer ordering, identical to
   cycle 206/213).
3. Build the float teacher: `teacher = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()`.
   Wrap teacher inference in `torch.no_grad()`. Do NOT delete after init eval — keep on device
   for the training loop.
4. For each seed in SEEDS=[42, 137, 2024]:
   a. `torch.manual_seed(seed); random.seed(seed)`.
   b. Build student: fresh `GPT2LMHeadModel.from_pretrained("gpt2")`, run `collect_mlp_io` on
      calib_ids, replace `model.transformer.h[b].mlp` with `FullCompressedMLP` from
      `fit_block_lstsq`. Move to device.
   c. Freeze ALL params; unfreeze `W_fc, b_fc, W_out, b_out` of the 3 compressed blocks.
   d. Adam optimizer LR=1e-4 on those params (matches cycle 206/213).
   e. Eval init PPL (sanity: ~61.7).
   f. Training loop: N_STEPS=10000, EVAL_EVERY=500. Iterate over `random.shuffle(calib_ids)`
      cycling indefinitely. At each step:
        - `with torch.no_grad(): teacher_logits = teacher(ids).logits`  # (B,T,V)
        - `student_logits = student(ids).logits`
        - `tau = 2.0`
        - `student_log_softmax = F.log_softmax(student_logits / tau, dim=-1)`
        - `teacher_softmax = F.softmax(teacher_logits / tau, dim=-1)`
        - `kd_loss = F.kl_div(student_log_softmax, teacher_softmax, reduction='batchmean') * (tau**2)`
        - `V = student_logits.size(-1)`
        - `ce_loss = F.cross_entropy(student_logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))`
        - `total_loss = 0.5 * kd_loss + 0.5 * ce_loss`
        - `optimizer.zero_grad(); total_loss.backward(); optimizer.step()`
        - For step ≤ 5: `print(f"  step {step}: kd={kd_loss.item():.4f} ce={ce_loss.item():.4f}")`
   g. At each EVAL_EVERY: `eval_ppl(student, eval_ids)`; record `(step, eval_ppl)`. Track
      `best_ppl, best_step` and snapshot a `copy.deepcopy(student.state_dict())` whenever a new
      best is observed.
   h. After training: restore best state via `student.load_state_dict(best_state)` and
      re-eval — store as `best_ppl_restored` (must equal `best_ppl` ± 1e-6).
5. Aggregate per-seed results: `init_mean/std`, `best_mean/std`, `best_step_mean`.
6. Verdict (item 229 thresholds):
   - `best_mean ≤ 44.0` → CONFIRMED
   - `best_mean ≥ 47.0` → REFUTED
   - else → INCONCLUSIVE
7. Write `autoresearch/results/cycle_224_distillation.json`:
   ```json
   {
     "cycle": 224, "item": 229, "verdict": "...",
     "config": {"BLOCKS":[1,4,7], "K":512, "N_STEPS":10000, "EVAL_EVERY":500,
                "LR":1e-4, "TAU":2.0, "KD_WEIGHT":0.5, "CE_WEIGHT":0.5,
                "SEEDS":[42,137,2024], "N_CALIB":200, "N_EVAL":100, "SEQ_LEN":128},
     "baselines": {"cycle_206_wout_only_ce": 48.111,
                   "cycle_210_wout_only_ce_4000_steps": 46.556,
                   "cycle_213_wout_wfc_ce": 46.424},
     "seeds": [{"seed":42,"init_ppl":...,"best_ppl":...,"best_ppl_restored":...,
                "best_step":...,"history":[...]}, ...],
     "aggregate": {"init_mean":...,"init_std":...,"best_mean":...,"best_std":...,
                   "best_step_mean":...}
   }
   ```
8. Write `autoresearch/results/cycle_224_summary.md` (verdict, table of seeds, comparison vs
   cycle 213/210/206 CE baselines).

# Validation

- Init PPL per seed ~61.75 (matches cycle 206 init; SEED only affects training shuffling+optim
  noise, not init since calib data is fixed).
- Float teacher PPL on eval set ~53.895 (cycle 206 float).
- KD loss must be > 0 throughout training; sanity-print first 5 step kd/ce values.
- Best-state restore round-trip: `abs(best_ppl_restored - best_ppl) < 1e-4`.
- best_step should land before N_STEPS in at least one seed (otherwise extend or note ceiling
  not yet reached).

# Artifacts

- experiments/distillation_baseline_cycle224.py
- autoresearch/results/cycle_224_distillation.json
- autoresearch/results/cycle_224_summary.md
- autoresearch/results/cycle_224_manifest.txt (written by execute agent)

# Fallbacks

- If 10000 × 3 seeds wall time exceeds budget: reduce to N_STEPS=5000 and record
  `config.reduced_steps_to=5000` in JSON. Cycle 213 W_out+W_fc reached its best at step 1500,
  so 5000 is plenty of headroom.
- If MPS dual-forward OOMs: process teacher and student forward sequentially (already
  sequential), and if needed chunk the batch dimension of `ids` (cycle 206 calib seqs are
  1×128 so this is unlikely to bite).
- If KD loss diverges (NaN/inf): clip grad norm to 1.0, then retry. If still diverging,
  fall back to pure KD (KD_WEIGHT=1.0, CE_WEIGHT=0.0) and document.
- The 44.0 threshold is aggressive: cycle 213 W_out+W_fc CE got 46.42 at the same parameter
  budget, so KD must improve by ≥2.4 PPL on a near-saturated optimum to confirm. Plausible
  outcome is INCONCLUSIVE.

# Update bookkeeping

- Move item 229 from NEXT/P2 to PAST/Cycle 224 in agenda.md (delete from NEXT block).
- Append cycle 224 row to autoresearch/results/cycles.tsv with verdict + best_mean.
- Update autoresearch/STATUS.md (latest cycle, current item, verdict line).
- Commit: `[autoresearch] cycle 224: KD distillation baseline (W_out+W_fc) — <verdict>`.
