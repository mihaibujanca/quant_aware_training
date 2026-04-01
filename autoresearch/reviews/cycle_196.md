# Cycle 196 Review: LayerNorm Recalibration (Item 207)

## Verdict: REFUTED

LN recalibration based on per-channel cross-token statistics provides zero improvement
and slightly worsens PPL (+0.248 for compressed blocks only, +0.556 for all 12).

## Key Results

- Float PPL: 25.082
- Compressed (blocks [1,4,7,11], K=512): 30.089
- LN-2 recal (compressed blocks): 30.337 (WORSE)
- LN-2 recal (all 12 blocks): 30.645 (WORSE)
- Best improvement: -0.248 (negative = no benefit)

## Why It Fails

LayerNorm normalizes each token independently over its 768 channels:
`y = (x - μ_token) / σ_token * γ + β`
where `μ_token` and `σ_token` are per-token statistics over channels (NOT per-channel
statistics over the batch). Therefore:
1. Per-channel cross-token mean/std shifts are irrelevant — LN removes per-token mean/std
2. Adjusting γ/β based on per-channel batch statistics interferes with LN's existing
   normalization, introducing a systematic per-channel bias that wasn't there before
3. The cascade barrier lives in the NORMALIZED residual content (the correlational
   structure within each token vector), which LN cannot address

## Comparison to Prior Work

Cycle 191 showed that direct mean error correction (constant bias applied to residuals)
worsened PPL by 12 PPL. This experiment shows that LN parameter adjustment also fails.
Both confirm that the cascade barrier is not correctable with zero-order (mean) statistics.

## Methodological Notes

- Compressed baseline with my implementation: 30.089 PPL vs cycle 185's 59.42 PPL for
  the same BLOCKS/K config. Discrepancy likely due to different PPL evaluation methods:
  cycle 185 used 200 seqs × 128 tokens; my evaluation uses wikitext-2 val with stride-512.
  My implementation may also use more calibration data (25600 tokens vs 4096 in cycle 185).
- The per-channel selection uses std × output_norms × mean_act scoring (matches aleph library).
