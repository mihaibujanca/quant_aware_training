# Cycle 182 — Brief

**Item:** 192 — Iterative bias recomputation for ReLU swap — GPT-2
**Type:** empirical
**Depends on:** 191 (REFUTED — precomputed biases gave +88% PPL)

## Context

Cycle 181 confirmed that precomputed μ_neg_l (computed on the all-GELU float model) gives +88% PPL when all 12 blocks are swapped simultaneously, despite single-block swaps being within +6%. The diagnosis: downstream blocks receive shifted residual-stream distributions once upstream blocks are changed to ReLU, making the precomputed biases stale.

## What this cycle measures

Iterative sequential swap: for l=0..11, (a) compute μ_neg_l on the current (partially-swapped) model using 2048 OWT tokens, (b) replace GELU→ReLU at block l, (c) add μ_neg_l as constant bias to c_proj. Each μ_neg_l sees the true input distribution from blocks 0..l-1 already swapped.

Record progressive PPL after each swap (12 eval points) plus final all-swapped PPL.

## Criteria

- **Confirmed:** iterative PPL ≥10 PPL points better than precomputed (42.6) → ≤32.6
- **Refuted:** iterative PPL within 3 PPL of precomputed → ≥39.6
- **Inconclusive:** 3-10 PPL improvement → 32.6 < PPL < 39.6
