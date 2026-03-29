# Cycle 096 Plan: Cascade Attribution — FFN vs Attention (GPT-2)

**Item:** 111
**Script:** `experiments/cascade_attribution_gpt2.py` (new)

## Pre-flights

**P1 -- Null hypothesis:** Cascade amplification is evenly split (40-60%) between FFN and attention sub-blocks. No single bottleneck exists; both contribute equally.

**P2 -- Criteria (verbatim from agenda):**
- Confirmed if: >=70% of cascade amplification is attributable to one sub-block type (FFN or attention) across >=3/4 compressed blocks.
- Refuted if: amplification is evenly split (40-60%) between FFN and attention (no single bottleneck).
- Inconclusive if: attribution varies by block position (early vs late).

**P3 -- Novelty vs replication:** Cycles 089-095 measured cascade rate c but never decomposed it by sub-block type. This is the first FFN-vs-attention attribution experiment.

**P4 -- Confounds:** Attention and FFN interact via the residual stream; "freezing" one sub-block's error is approximate (error from sub-block A affects sub-block B within the same block). Mitigated: freeze-both ablation isolates the interaction term (I1 from review). Block 0 attention sensitivity (095) may dominate -- mitigate by reporting per-block breakdown.

**P5 -- Cost estimate:** k=4 compressed blocks x 4 ablation conditions x 3 seeds x 200 eval seqs = 48 evals + 12 instrumented forward passes. ~25 min on MPS.

**P6 -- Data dependencies:** `data/openwebtext_sample.txt` (same corpus as cycles 089-095).

**P7 -- Abort conditions:** Baseline PPL outside [25, 65]. Any ablation PPL > 50x baseline (diverged).

## Implementation

1. **Load GPT-2**, baseline PPL on OWT, 200 eval seqs, 3 seeds [42, 123, 7].
2. **Apply k=4 FFN compression** using topk + lstsq + 500 Adam polish at 6x (same config as cycle 089, forward order [7, 5, 4, 1]).
3. **Descriptive: instrument residual stream (norm-based, NOT causal).** For each block L, hook after attention (post-attn residual) and after FFN (post-FFN residual). Record:
   - `delta_attn[L]` = L2 norm of (compressed_post_attn - float_post_attn), averaged over eval seqs
   - `delta_ffn[L]`  = L2 norm of (compressed_post_ffn - float_post_ffn), averaged over eval seqs
   - `amplification_attn[L]` = delta_attn[L] / input_error[L] (how much attention amplifies incoming error)
   - `amplification_ffn[L]`  = delta_ffn[L] / delta_attn[L] (how much FFN amplifies post-attn error)

   These are **descriptive diagnostics** — they measure norm deltas but mix locally-generated and propagated error. They do NOT establish causal attribution. The causal test is the PPL ablation below.

4. **Causal ablation — 4 conditions (global, all compressed blocks simultaneously).**
   - **Full compressed**: no intervention. Record `full_compressed_ppl`.
   - **Ablation A — freeze attention error**: in all 4 compressed blocks, after attention, replace output with float attention output (kill attention error, keep FFN error). Record `freeze_attn_ppl`.
   - **Ablation B — freeze FFN error**: in all 4 compressed blocks, after FFN, replace output with float FFN output (kill FFN error, keep attention error). Record `freeze_ffn_ppl`.
   - **Ablation C — freeze both**: in all 4 compressed blocks, replace both attention AND FFN output with float outputs (kill all compression error in these blocks). Record `freeze_both_ppl`.

   Supplementary: run per-block ablations (freeze in one block at a time) for positional resolution, but do NOT use these for the verdict.

5. **Attribution (no assumption that attn% + ffn% = 100%).** Define `D = full_compressed_ppl - baseline_ppl` (total degradation). Report all four raw PPL values, then decompose:
   ```
   attn_contribution  = full_compressed_ppl - freeze_attn_ppl     # PPL drop when removing attn error
   ffn_contribution   = full_compressed_ppl - freeze_ffn_ppl      # PPL drop when removing FFN error
   both_contribution  = full_compressed_ppl - freeze_both_ppl     # PPL drop when removing all error
   interaction        = both_contribution - attn_contribution - ffn_contribution
                      # = full - freeze_attn - freeze_ffn + freeze_both
   ```
   Report: `attn_contribution / D`, `ffn_contribution / D`, `interaction / D`. These three sum to `both_contribution / D` (which may be < 1.0 if non-compressed blocks also contribute cascade error).

6. **Verdict logic:**
   ```
   attn_frac = attn_contribution / D
   ffn_frac  = ffn_contribution / D
   dominant = "attn" if attn_frac > 0.70 else "ffn" if ffn_frac > 0.70 else None
   # Per-block supplementary: check consistency
   consistent = sum(1 for L in compressed_blocks if per_block_dominant[L] == dominant) >= 3
   if dominant and consistent: CONFIRMED (dominant sub-block identified)
   elif 0.40 <= attn_frac <= 0.60 and 0.40 <= ffn_frac <= 0.60: REFUTED
   else: INCONCLUSIVE
   ```
7. **Output:**
   - JSON: `autoresearch/results/cycle_096_cascade_attribution.json` — must include all 4 PPL values (full_compressed, freeze_attn, freeze_ffn, freeze_both), derived contributions, interaction term, per-block supplementary.
   - Plot: stacked bar chart (4 bars, one per compressed block), showing attn vs FFN vs interaction attribution fraction. Save as `..._attribution.png`.
8. **Agenda update + commit** per standard autoresearch flow.
