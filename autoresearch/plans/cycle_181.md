# Cycle 181 — Plan

**Item:** 191 — Full-model ReLU + per-block bias vector swap — GPT-2

## Pre-flights

1. Load GPT-2, eval baseline PPL on 200 OWT sequences (expect ~22.7). Abort if outside [15, 65].
2. Verify model has 12 blocks with GELU activation accessible at `model.transformer.h[b].mlp.act`.

## Design

### Phase 1 — Precompute μ_neg_l (2048 tokens)

For each block l ∈ {0..11}:
1. Hook `model.transformer.h[l].mlp.act` to capture pre-activation z and post-activation GELU(z).
2. Compute y_neg = GELU(z) ⊙ 1[z ≤ 0] per token (the negative-regime contribution in 3072-dim).
3. Accumulate E[y_neg] over 2048 tokens (first ~2 sequences of 1024 tokens each).
4. Project through W_down: μ_neg_l = W_down @ E[y_neg] (768-dim vector per block).

This is equivalent to E[W_down @ y_neg] by linearity. Sanity check: compare ||μ_neg_l|| across blocks.

### Phase 2 — Full-model swap

For all 12 blocks simultaneously:
1. Replace `model.transformer.h[l].mlp.act` with a module that computes ReLU(z) instead of GELU(z).
2. Add μ_neg_l to the block's c_proj bias: `model.transformer.h[l].mlp.c_proj.bias += μ_neg_l`.

This makes the modified block compute: y_l = W_down @ ReLU(z) + b_down + μ_neg_l.

### Phase 3 — Evaluation

Eval full-model PPL on 200 OWT sequences (same eval set as baseline).

Also measure per-block incremental swap (blocks 0..k for k=0..11) to see cascade accumulation pattern.

## Measurements

- `ppl_base`: float GELU baseline
- `ppl_full`: all 12 blocks swapped simultaneously
- `ppl_incremental[k]`: blocks 0..k swapped, rest float
- `mu_neg_norms[l]`: ||μ_neg_l|| for each block
- `delta_pct`: (ppl_full - ppl_base) / ppl_base * 100

## Verdict logic

- delta_pct <= 10 → CONFIRMED
- delta_pct > 50 → REFUTED
- 10 < delta_pct <= 50 → INCONCLUSIVE

## Script

`experiments/full_relu_bias_swap_cycle181.py`

## Output

- `autoresearch/results/cycle_181_relu_bias_swap.json`
- `autoresearch/results/cycle_181_relu_bias_swap.png`
