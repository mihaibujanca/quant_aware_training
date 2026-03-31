# Cycle 166 — Item 176: Functional role of constant negative bias

## Context
Cycle 165 confirmed negativity is **learned** (W_up-driven, not random): W_up @ x mean ranges -0.59 to -1.06 across blocks, bias contributes only 8-10%. Cycle 164 confirmed positive channels are **essential**: killing them → PPL 713K (per-block cascade 970K at block 0). Current hypothesis: negative regime is structurally sparse + token-invariant, possibly just a learnable bias.

## Item 176 Test Plan
Test if negative regime functions as a **constant bias offset**:
- (a) Compute b_leak = E[y_neg] (mean output of negative regime) per block on 4096 tokens
- (b) Modify GPT-2: replace GELU(z) → ReLU(z) (kills all negative), add b_leak as fixed constant to block output
- (c) Measure per-block and full-model PPL of "ReLU + b_leak" vs float baseline

## Pass/Fail Criteria
- **CONFIRMED**: ReLU + b_leak recovers to within +5% PPL in ≥8/12 blocks → negative regime is functionally just bias
- **REFUTED**: >+30% PPL degradation → token-dependent signal is essential
- **INCONCLUSIVE**: +5-30% PPL

## Dependency
None. ~10 min compute.

## Key Dependencies from Prior Work
- Positive channels indispensable (cycle 164: +970K% block 0)
- Negativity is learned weight-data alignment (cycle 165: -0.99 to -1.03 mean)
- Token-dependence: 87-98% of y_neg is constant, 2-13% varies by token

## Notes
If confirmed: negative regime = 768 params/block of redundancy in compression.
If refuted: understand why token-variance matters despite >85% constancy.
