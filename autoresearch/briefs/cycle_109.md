# Cycle 109: Negative Leak Decomposition — GPT-2 FFN

**Item 120** [INTERPRETABILITY]

## Hypothesis
Cycle 103 observed 53-64% of bump-zone activations are z < 0, increasing with depth. GELU's "negative leak" (small signed output for z < 0, vs ReLU's zero) may be mostly constant bias correction across tokens, not per-token computation.

## Task
Decompose negative leak on 4096 mixed-corpus tokens across all 12 blocks:
- Partition tokens per channel: z_j > 0 (positive) vs z_j < 0 (negative)
- Compute: μ_leak_j = E[W_out[:,j] · GELU(z_j) | z_j < 0]
- Compute: Var and energy fraction E_neg / (E_neg + E_pos) per channel
- Aggregate per block: bias term b_leak = Σ_j μ_leak_j, token-dependent signal s_leak(x)
- Measure: ||b_leak|| / ||mean_block_output|| and ||s_leak|| / ||block_output||
- For top-20 variance leak channels: semantic patterns in extreme tokens?

## Success Criteria
- **Confirmed**: b_leak accounts for >70% negative-regime energy in ≥8/12 blocks (leak is systematic shift)
- **Refuted**: s_leak accounts for >70% in ≥8/12 blocks (leak computes per-token signal)
- **Inconclusive**: 30-70% split or variable across blocks

## Context
Builds on cycle 103 (bump-zone anatomy). Connects to cycle 108 (pre-activation distribution). Core question: is negative leak a trainable constant or active computation?
