# Cycle 113 Plan: Per-Head Amplification Census — GPT-2

**Item:** 124 | **Script:** `experiments/per_head_amplification_census_gpt2.py` (new)

## Criteria (verbatim from agenda)
- **Confirmed if:** top 3 heads per layer account for >=70% of total attention-mediated amplification in >=8/12 layers.
- **Refuted if:** amplification uniformly distributed (<=15% max per head) in >=8/12 layers.
- **Inconclusive if:** concentrated in some layers but not others, or top-3 at 40-70%.

## Pre-flights
**P1 -- Null:** If all heads have equal amplification, each head = 1/12 = 8.3%, top-3 = 25%. Confirmed threshold (70%) is well above null. Refuted threshold (15%) is ~2x null. Both are informative.
**P2 -- Cost:** 12 compressed blocks x 1 seed = 12 FFN fits + 12 per-head measurements. ~15 min MPS.
**P3 -- Abort:** Baseline PPL outside [25, 65]. Any block compression fails (output NaN).
**P4 -- Data:** OWT, 256 seqs (2048+ tokens per seq not needed: aggregate over token dim).
**P5 -- Prior:** Cycle 104: attention contributes >=70% of cascade amplification in 8/11 transitions.

## Implementation

1. **Baseline PPL.** Load GPT-2, eval 200 seqs. Abort if outside [25, 65].

2. **Per-block FFN compression.** For each block L in 0..11:
   Compress FFN at 6x (k=512, topk+lstsq+500 Adam) using existing `derive_ffn_compression`
   + `swap_compressed_ffns`. Store compressed model variant.

3. **Collect float and perturbed residual streams.** For each compressed block L,
   run float model and compressed model on eval data. Hook the output of block L
   (residual stream entering block L+1). Compute delta_L = x_compressed - x_float.
   For L=11 (last block), skip — no downstream attention to measure.

4. **Per-head amplification measurement.** For each block L in 0..10 (11 transitions):
   Focus on block L+1's attention layer. For each head h in 0..11:
   - Hook the head's output projection (isolate head h's contribution).
   - GPT-2 attention: `attn_output = sum_h (softmax(Q_h K_h^T / sqrt(d)) V_h) W_O_h`
   - Compute head_out_float_h = Attn_h(LN(x_float)) and head_out_perturbed_h = Attn_h(LN(x_float + delta_L))
   - Per-head amplification: amp_h = mean_tokens(||head_out_perturbed_h - head_out_float_h||_2) / mean_tokens(||delta_L||_2)
   **Key:** feed perturbed input through LN_1 of block L+1, then decompose attention output by head.

5. **Concentration metrics.** Per layer L+1:
   - top3_frac = sum(top 3 amp_h) / sum(all amp_h)
   - max_frac = max(amp_h) / sum(all amp_h)
   - Gini coefficient across 12 heads.

6. **Ablation importance (optional, low cost).** For each head h, zero out c_proj
   rows for that head, measure PPL delta. Compare with amplification ranking via
   Spearman correlation. Run once (no seeds needed — deterministic).

7. **Verdict logic.**
   - layers_concentrated = count(layers where top3_frac >= 0.70), over 12 layers (L+1 = 1..11, plus L+1=0 measuring block 0 attn with delta from... skip: only 11 transitions L=0..10).
   - layers_uniform = count(layers where max_frac <= 0.15).
   - If layers_concentrated >= 8: CONFIRMED.
   - If layers_uniform >= 8: REFUTED.
   - Else: INCONCLUSIVE.

8. **Output.**
   - JSON: `autoresearch/results/cycle_113_head_amplification.json`
   - Heatmap: 12x12 amplification factor (rows=downstream layer, cols=head), `autoresearch/results/cycle_113_heatmap.png`
   - Bar chart: top-10 amplifying heads across model.
   - Manifest: `autoresearch/results/cycle_113_manifest.txt`
