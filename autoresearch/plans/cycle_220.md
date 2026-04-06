# Task
Item 234 (P0): Surgical bump removal. Isolate GELU's negative leak from its positive-zone bump by patching all 12 GPT-2 small MLP activations with two surgically-defined variants and measuring full-model PPL on the shared eval set.

# Hypothesis
Claim: "The negative leak and positive-zone bump contribute unequally to model performance. One of them is functionally dominant; the other is near-linearizable."

Criteria (verbatim from item 234):
- Confirmed if: one of {GELU_noneg, GELU_nobump} all-12 PPL ≤ 25 (within 10% of float) AND the other ≥ 35 (>55% degraded) — clean separation.
- Refuted if: both variants degrade similarly (within 3 PPL of each other) — the two sub-regions are entangled.
- Inconclusive if: both degrade 20-50% with no clean separation.

# Pre-flight Results
- P1 duplicate: cycles 184 (ReLU+bias), 188 (C3 mean), 190 (GELU_neg+ReLU_pos hybrid), 193 (C3+Hybrid) measured related variants. NONE tested GELU_noneg or GELU_nobump as defined here. NEW.
- P2 dependency: none.
- P3 ≤30 min: yes — 3 forward passes, no training, ~30 s wall time.
- P4 program/prompts: no changes.

# Steps
1. Create `experiments/surgical_bump_removal_cycle220.py`.
2. Reuse cycle 206's loader verbatim: wikitext-2 validation, tokenizer pad=eos, SEQ_LEN=128, build all_ids, keep N=200 evaluation sequences. Seed 42.
3. Copy cycle 206's `eval_ppl` pattern: forward pass with labels=input_ids, accumulate token-weighted CE, return exp(mean_ce). Use `torch.no_grad()` and `model.eval()`.
4. Define activations as nn.Modules wrapping a function:
   - `GeluNoneg`: `torch.relu(F.gelu(z, approximate='tanh'))`.
   - `GeluNobump`: piecewise, implemented with masks to stay autograd-safe and vectorized:
     `hi = (z > 2).float(); lo = (z < -2).float(); mid = ((z >= -2) & (z <= 2)).float()`
     `return hi * z + mid * (0.5 * (z + 2)) + lo * 0`
     Continuity: z=2 → 0.5·4 = 2 = z ✓; z=-2 → 0 ✓.
5. Sanity-print 5 anchor values for each variant on z ∈ {-3, -1, 0, 1, 3} before patching. Compare baseline GELU, gelu_noneg, gelu_nobump.
6. Patch: for each variant, save original modules via
   `originals = [model.transformer.h[i].mlp.act for i in range(12)]`
   then `model.transformer.h[i].mlp.act = variant_module()` for all 12 blocks. After eval, restore originals. Verify after restore with a sanity PPL equal to baseline.
7. Evaluate PPL for: `baseline`, `gelu_noneg`, `gelu_nobump`. Record baseline first, restore, patch noneg, eval, restore, patch nobump, eval, restore, re-eval baseline sanity.
8. Compute deltas:
   - `noneg_delta = noneg_ppl - baseline_ppl`; `noneg_pct = noneg_delta / baseline_ppl * 100`
   - same for nobump.
9. Verdict logic — RELATIVE thresholds derived from in-cycle baseline (design-review revision):
   ```
   low_thresh = baseline_ppl * 1.10   # within 10% of float (the agenda's "≤25" anchor)
   high_thresh = baseline_ppl * 1.55  # >55% degraded (the agenda's "≥35" anchor)
   if (noneg_ppl <= low_thresh and nobump_ppl >= high_thresh) or (nobump_ppl <= low_thresh and noneg_ppl >= high_thresh):
       verdict = "CONFIRMED"
   elif abs(noneg_ppl - nobump_ppl) <= 3:
       verdict = "REFUTED"
   else:
       verdict = "INCONCLUSIVE"
   ```
   Log low_thresh and high_thresh into the JSON alongside baseline_ppl.
10. Write `autoresearch/results/cycle_220_surgical_bump.json`:
    ```
    {"cycle": 220, "item": 234, "verdict": ...,
     "baseline_ppl": ..., "gelu_noneg_ppl": ..., "gelu_nobump_ppl": ...,
     "noneg_delta": ..., "noneg_pct": ...,
     "nobump_delta": ..., "nobump_pct": ...,
     "low_thresh": ..., "high_thresh": ...,
     "prior_baselines": {"cycle_184_relu_bias_pct": 87.7,
                         "cycle_188_c3_mean_pct": 20.33,
                         "cycle_190_hybrid_pct": 27.67},
     "config": {"N_EVAL": 200, "SEQ_LEN": 128, "SEED": 42, "DEVICE": "..."}}
    ```
11. Write `autoresearch/results/cycle_220_summary.md` — 1 sentence plain-language conclusion plus the 4-point table (baseline, hybrid[190], GELU_noneg, GELU_nobump) with PPL and %.

# Validation
- baseline_ppl matches historical ~22-26 (cycle 184 measured 22.7).
- Activation sanity print shows: GELU(-3)≈-0.0036, gelu_noneg(-3)=0, gelu_nobump(-3)=0; GELU(3)≈2.9960, gelu_noneg(3)≈2.9960, gelu_nobump(3)=3. Differences are meaningful (not 0.01 PPL).
- Post-restore baseline re-eval equals first baseline (confirms patch/restore is clean).

# Artifacts
- `experiments/surgical_bump_removal_cycle220.py`
- `autoresearch/results/cycle_220_surgical_bump.json`
- `autoresearch/results/cycle_220_summary.md`
- `autoresearch/results/cycle_220_manifest.txt` (written by execute agent)

# Fallbacks
- If `model.transformer.h[i].mlp.act` is not a swappable attribute (HF NewGELUActivation wrapping differs), fall back to a forward pre-hook on `mlp.c_proj` that replaces its input using the variant applied to the output of `mlp.c_fc`. Reference cycle 217's hook pattern.
- If baseline PPL deviates from cycles 184/188/190 baseline by >1 PPL, document eval-set construction difference in the summary; verdict still uses the in-cycle baseline.
- If MPS autograd misbehaves for the masked nobump, fall back to `torch.where` chains on CPU for the patched activation (forward only, no grads needed).

# Update bookkeeping
- Move item 234 from NEXT/P0 to PAST/Cycle 220 in `autoresearch/agenda.md`.
- Append a TSV row to the cycle log.
- Commit: `[autoresearch] cycle 220: surgical bump removal — GELU_noneg vs GELU_nobump`.
- Refresh `autoresearch/STATUS.md`.
