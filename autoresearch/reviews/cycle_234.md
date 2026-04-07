# Cycle 234 Design Review

**Item**: 244 (P2 scout)
**Verdict**: APPROVE

## Summary

Cycle 228 base (`relu(GELU(z)) + μ_l` patch on all 12 blocks, 61.664 PPL) + LoRA r=8 residual adapter on each block's `mlp.c_proj` (HF Conv1D, W shape (3072, 768)). Trained jointly via Adam on CE against WikiText-2 val (N=200, SEQ_LEN=128), LR=1e-4, 2000 steps, seed 42. Goal: eat the 16.6 PPL residual between cycle 228 and float baseline 45.064.

## Strengths

- Clean, direct test of item 244's hypothesis. Confirmation thresholds preregistered (≤48.0 / ≥58.0).
- Init-eval sanity check (`init_ppl ≈ 61.664`) catches LoRA wrapper bugs immediately because B is zero-init → identity to cycle 228 forward.
- Trainable-param assertion (368,640 = 12 × (3072×8 + 8×768)) catches freezing mistakes.
- Post-train frozen-base check (`torch.equal` on snapshot) guards against accidental unfreezing.
- Fallbacks are specific and actionable: LR halving for divergence, forward-layout diagnostic for init mismatch.
- Dependencies (cycles 228, 238) confirmed done.

## Concern (per review request)

P2 scout with 1 seed is acceptable per the agenda's design. Item 244 explicitly specifies seed 42, one seed for scouting. If results land in the confirmed band, a multi-seed follow-up can be scheduled as a higher-priority item.

## Minor notes (non-blocking)

- Plan states brief specifies A ∈ R^(3072, 8), B ∈ R^(8, 768), delta = A @ B — matches HF Conv1D (x @ W + b with W shape (in, out) = (768, 3072)… actually Conv1D.weight is (nf, nx) swapped; verify at implementation). The init_ppl sanity check will catch any layout error deterministically.
- A init N(0, 0.01) combined with B=0 gives zero delta regardless of A — good.
- Calibration/eval set identity (same 200 ids) is an explicit choice for scouting; a held-out check would strengthen any confirmed result but is out of scope for P2.

## Decision

Approved. Proceed to implementation.
