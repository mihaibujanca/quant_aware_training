# Cycle 178 — Brief

**Item:** 188 — Negative-regime bias absorption + positive-only compression — GPT-2 FFN k=4
**Type:** empirical
**Depends on:** none (uses cycle 166/173 bias computation + existing topk)

## Context

Cycles 164-175 established that the GELU negative regime is a constant bias per block:
- Cycle 173 (C3 condition): replacing GELU negative output with per-channel conditional mean E[gelu(z)|z<0] costs at most +1.4% PPL per block (12/12 within +2%).
- Cycle 166: ReLU+b_leak per block costs +0.8-5.9% PPL; full-model cascade is +87%.
- ~87-91% of intermediate channels are typically in the negative regime.

The hypothesis: if we absorb the constant negative contribution as a bias and zero the negative channels (GELU->ReLU), the effective hidden dimension for compression drops from 3072 to ~270-400 positive-regime channels. Topk+lstsq at k=4 (512 channels) on this reduced set should compress better because the lstsq problem is better conditioned.

## What this cycle measures

1. **Absorption-only baseline**: all 12 blocks with GELU->ReLU+b_leak, no compression. PPL.
2. **Positive-only topk+lstsq**: 4 blocks [1,4,7,11] with bias absorption + topk on positive-regime channels + lstsq. PPL.
3. **Standard topk+lstsq** (control): same 4 blocks, standard GELU, topk k=4 (512 channels) + lstsq + 500-step polish.

## Criteria

- **Confirmed:** positive-only compression PPL <= 68 at same or fewer params as standard topk+lstsq
- **Refuted:** absorption cascade >5 PPL degradation before compression, or positive-only worse than full topk+lstsq
- **Inconclusive:** absorption stable but positive-only ~ full topk+lstsq
