# Cycle 107: Output-Space Direct Regression Bypass — GPT-2 FFN

**Item**: 117
**Status**: PENDING
**Date**: 2026-03-29

## Context

All hidden-layer compression approaches have failed:
- Jacobian factorization (100): full-rank, J(x)@x ≠ FFN(x) for GELU
- GELU clustering (101): K* > 256 across blocks
- Subspace projection (99): error spans 85% of dimensions

**Core idea**: Bypass hidden layer entirely. For each FFN block, fit direct low-rank input-output regression f(x) ≈ U @ S @ V^T @ x using lstsq, no GELU or activation patterns.

## Experiment Plan

Sweep rank r ∈ {32, 64, 128, 256, 384} on single-block swaps at blocks [1, 4, 7, 11].
Use 4096 OWT probe tokens for lstsq fitting.

**Parameter matching**:
- Direct rank-r: 768×r×2 = 1536r params
- Topk k: 768k + k×768 = 1536k params (matches directly)

**Hypothesis**: If hidden-layer structure is a compression liability, direct regression should match or beat topk+lstsq at matched param count.

## Success Criteria

- **Confirmed**: Direct rank-r beats topk at matched params in ≥2/4 blocks
- **Refuted**: Topk outperforms by ≥5% PPL across all matched comparisons
- **Inconclusive**: Block-dependent or within 5%

## References

- Cycle 104: Cascade source CONFIRMED — attention dominates error
- Cycle 106: k=5 polish +22-24% PPL (not viable vs +10% threshold)
