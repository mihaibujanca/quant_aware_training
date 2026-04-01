# Cycle 182 — Plan

**Item:** 192 — Iterative bias recomputation for ReLU swap — GPT-2

## Pre-flights

1. Verify `data/openwebtext_sample.txt` exists
2. Verify `gpt2` model loads and baseline PPL in [15, 65]
3. Verify cycle 181 result file exists (precomputed baseline: PPL 42.643)

## Method

Script: `experiments/iterative_relu_bias_swap_cycle182.py`

Base the script on cycle 181's `full_relu_bias_swap_cycle181.py`. Key change:

### Sequential iterative loop (l=0..11):
1. Hook block l's activation (GELU) to collect GELU(z) ⊙ 1[z≤0] on 2048 tokens
2. Compute μ_neg_l = W_down @ E[y_neg] — using current model state (blocks 0..l-1 already ReLU)
3. Replace block l's GELU with ReLU
4. Add μ_neg_l to block l's c_proj bias
5. Eval PPL on 200 OWT sequences → ppl_progressive[l]
6. Do NOT restore — keep the swap for next iteration

After all 12: final PPL = ppl_progressive[11].

### Comparison
- Precomputed PPL from cycle 181: 42.643
- Float baseline: 22.7
- Delta = precomputed PPL - iterative PPL (should be positive if iterative is better)

### Output
- `autoresearch/results/cycle_182_iterative_relu_swap.json`
- `autoresearch/results/cycle_182_iterative_relu_swap.png`
- Plot: progressive PPL curve (iterative) vs cycle 181 incremental curve

### Verdict logic
- `abs_improvement = 42.643 - iterative_final_ppl`
- CONFIRMED if abs_improvement >= 10
- REFUTED if abs_improvement <= 3
- INCONCLUSIVE if 3 < abs_improvement < 10

## Risks

- MPS eval may be slow (12 × 200 seq evals). ~15 min expected.
- μ_neg estimation on 2048 tokens may be noisy for later blocks. Accept this — matching item 191 config.
