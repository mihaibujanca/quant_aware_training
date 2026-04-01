# Cycle 180 — Plan

**Item:** 190 — FLAP-style bias compensation for topk pruning — GPT-2 FFN k=4

## Pre-flights

1. Load GPT-2, eval baseline PPL on 200 OWT sequences (expect ~22.7). Abort if outside [15, 65].
2. Probe FFN geometry for blocks [1,4,7,11] using existing `probe_ffn_geometry`.
3. Verify `hidden_sample` (post-GELU activations) is available in atlas per block.

## Design

### Condition A — Analytic bias only (no lstsq)

For each compressed block b in [1,4,7,11]:
1. Run topk selection → get `selected` (512 indices) and `removed` (2560 indices).
2. Build CompressedMLP with k=512 hidden units.
3. c_fc weights/bias: original W_up[selected], b_up[selected] (same as baseline).
4. c_proj weight: original W_down[:, selected] (the original columns, NOT lstsq-fitted).
5. c_proj bias: original b_down + sum_j_in_removed( E[W_down[:,j] * GELU(z_j)] ).
   - E[GELU(z_j)] is computed as mean of hidden_sample[:,j] over calibration tokens.
   - The bias correction vector is W_down[:, removed] @ mean_hidden[removed] (768-dim).

### Condition B — topk + lstsq (control)

Standard: `_fit_output_projection(hidden[:,selected], output, use_bias=True)`.

### Condition C — Analytic bias + polish

Condition A model, then 500 steps Adam lr=1e-4 on c_proj weights+bias (same as baseline polish).

### All conditions

- 3 seeds {42, 137, 7}, 200 OWT eval sequences, blocks [1,4,7,11], k=512
- Measure: stacked PPL for each condition

## Measurements

Per seed:
- `ppl_base`: float baseline
- `ppl_A_pre`: analytic bias, no polish
- `ppl_B_pre`: topk+lstsq, no polish
- `ppl_B_post`: topk+lstsq + 500 polish
- `ppl_C_post`: analytic bias + 500 polish
- Per-block: lstsq residual MSE (B) vs original-columns residual MSE (A)

## Verdict logic

- Mean ppl_A_pre <= 73 OR mean ppl_C_post <= 68.7 → CONFIRMED
- Mean ppl_A_pre > 85 AND |mean ppl_C_post - mean ppl_B_post| <= 1 → REFUTED
- Otherwise → INCONCLUSIVE

## Script

`experiments/flap_bias_compensation_cycle180.py`

## Output

- `autoresearch/results/cycle_180_flap_bias.json`
