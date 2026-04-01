# Cycle 195 Brief: Jacobian Effective Rank Census (Item 206)

## Item

206 — Jacobian effective rank census — data-active rank of GPT-2 FFN blocks

## Context

The program (R3) mandates measuring the data-active geometry of FFN blocks using
MASO/spline theory tools. The core question: what is the effective rank of the per-token
Jacobian J(x) = ∂f_block(x)/∂x for each FFN block, measured on real data?

If Jacobians are low-rank (<100/768), the block's function is concentrated in a small
active subspace — a strong geometric compression signal. If full-rank (>300/768), the
block distributes its function across many directions, confirming the diffuse structure
found in cycles 184/189/194.

## Experiment Design

Use `torch.func.jacrev` to compute per-sample Jacobians for each FFN block.
Each Jacobian: 768×768 (input dim → output dim).

Config: N=50 seqs, SEQ_LEN=32 (small for compute), batch_size=1.
This gives 50×32=1600 per-block Jacobians.

Compute:
1. Effective rank at 90% energy of each Jacobian
2. Mean effective rank and variance across tokens per block
3. Rank of the STACKED Jacobian matrix (all tokens concatenated)

## Notes

- torch.func.jacrev requires careful handling on MPS (may need CPU fallback)
- Full 768×768 Jacobians for 1600 tokens × 12 blocks is compute-heavy
- Use per-sample forward-mode autodiff if available
- Short sequences to stay within 30-minute cap
