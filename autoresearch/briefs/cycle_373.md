# Cycle 373 Brief — Item 381

**Task:** Probe FFN local Jacobians on calibration data — data-active compression feasibility (R3)

## Context
- **Working recipe**: topk+lstsq+MLP adapter, 4.79x per-block compression, PPL 47.43 on 3-block joint, PPL 65.54 on 6-block (composition barrier).
- **Composition barrier**: Static weight compression works locally (best: 46.22 PPL single-block) but collapses at 6+ blocks (PPL 65.54). R3 direction seeks to compress data-active local linear maps (per-token Jacobians) rather than static weights.
- **Success metric**: If mean Jacobian eff_rank < 0.5 × weight eff_rank → data-active compression has headroom over static compression.
- **Failure metric**: If Jacobian eff_rank >= weight eff_rank → data conditioning doesn't help.

## Measurement Plan
1. Compute per-token Jacobians J = dFFN/dx on 512 calibration tokens for blocks b1, b5, b11
2. Report: (1) mean Jacobian eff_rank, (2) variance of rank across tokens, (3) top-10 singular value concentration
3. Compare to weight matrix SVD rank
4. Determine if Jacobian rank << weight rank (feasible for data-active compression) or ≥ (static compression already near-optimal)

## Recent Context
- Cycle 372: Importance-weighted k allocation → PLAN INVALID (infrastructure issue)
- Cycle 371: Residual stream geometry across compression stages → INCONCLUSIVE (PPL barrier real but not visible in eff_rank/topk/cosine)
- Cycle 370: Selective compression (b10+b11 uncompressed) → REFUTED (composition barrier is global, not block-specific)

## Technical Note
Jacobian computation: use `torch.func.jacrev` on FFN forward, shape (n_tokens, d_in, d_out). Effective rank via nuclear norm / Frobenius ratio.
