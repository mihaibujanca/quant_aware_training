# Cycle 104 Plan: Cascade Error Source — Residual vs Nonlinearity (GPT-2 FFN)

**Item:** 115
**Script:** `experiments/cascade_error_source_gpt2.py` (new)
**Revision:** 1 (fixes I1, I2; addresses A1, A2)

## Pre-flights

**P1 -- Null hypothesis:** No dominant source; attention and GELU each contribute 40-60%.
**P2 -- Criteria:** Confirmed if one source ≥70% excess in ≥8/11 transitions. Refuted if all 40-60%. Else inconclusive.
**P3 -- Cost:** 11 transitions × 3 seeds × 4 variants = 132 fwd passes. ~12 min MPS.
**P4 -- Abort:** Baseline PPL outside [25, 65]. Any ablation PPL > 5× baseline.

## Implementation

1. **Baseline.** Load GPT-2, eval PPL on `data/openwebtext_sample.txt`, 200 seqs, seeds [42, 123, 7].

2. **Single-block compression.** For each source block L in 0..10, compress block L FFN (topk+lstsq+500 Adam, 6×). Record δ_L = hidden_compressed − hidden_float at block L output (post-FFN, post-residual-add, pre-LN).

3. **Four-variant propagation through block L+1.** Feed compressed residual into block L+1:
   - **Full (F):** Normal forward. Measure δ_F at block L+1 output.
   - **Attn-frozen (A):** Attention sub-layer receives float LN output; FFN sees perturbed residual. Measure δ_A.
   - **FFN-frozen (B):** FFN sub-layer receives float post-attn residual; attention sees perturbed residual. Measure δ_B.
   - **Both-frozen (D):** Both sub-layers receive float inputs; only residual stream carries error. Measure δ_D.

4. **Decomposition (excess over passthrough).** Per transition, per seed:
   ```
   amp_full = ||δ_F|| - ||δ_D||          # total excess amplification
   amp_attn = ||δ_B|| - ||δ_D||          # attention-only excess
   amp_gelu = ||δ_A|| - ||δ_D||          # FFN-only excess
   amp_inter = amp_full - amp_attn - amp_gelu  # interaction
   attn_frac = amp_attn / amp_full
   gelu_frac = amp_gelu / amp_full
   inter_frac = amp_inter / amp_full      # fractions sum to 1.0
   ```
   Sanity: assert `attn_frac + gelu_frac + inter_frac ≈ 1.0` per row.

5. **LayerNorm passthrough contribution (A2).** Report ||δ_D|| / ||δ_L|| per transition. If ratio ≠ 1.0, LN itself amplifies/attenuates. Log in results JSON.

6. **Cosine tracking (A1).** Compute `cos(δ_F, δ_L)` per transition — measures how much each sub-layer rotates the error vector. Report alongside norm fractions.

7. **Verdict logic.** Average fractions across seeds per transition.
   ```
   dominant_count = sum(1 for t if max(attn_frac[t], gelu_frac[t]) >= 0.70)
   same_source = all dominant labels identical
   if dominant_count >= 8 and same_source:       CONFIRMED (name source)
   elif all(0.40 <= attn_frac[t] <= 0.60):       REFUTED
   else:                                          INCONCLUSIVE
   ```

8. **Output:**
   - JSON: `autoresearch/results/cycle_104_cascade_source.json` — per-transition: all four norms, fractions, LN ratio, cosine, verdict.
   - Plot: stacked bar (11 bars) for attn / gelu / interaction fractions. Second y-axis: LN passthrough ratio. Save `autoresearch/results/cycle_104_cascade_source.png`.

9. **Agenda update + commit** per standard autoresearch flow.
