# Cycle 198 Brief: Block 0 Criticality Analysis (Item 205)

## Item

205 — Block 0 criticality analysis — what makes block 0 unique

## Context

Cycle 187 found: dropping block 0 gives PPL 1923 vs all other blocks <41.
Block 0 handles raw token embeddings (first transformer block). Why is it so critical?

## Experiment Design

Analyze block 0's geometry vs blocks 1-11:
1. Activation covariance rank (rank_90 of hidden activations' covariance matrix)
2. Channel utilization: fraction of channels consistently active (|GELU(z_c)| > 0.1)
3. Output norm distribution: ||block_0_output - input|| vs other blocks
4. Weight singular value concentration: how concentrated is the spectrum?

For each metric, compare block 0 to blocks 1-11 to identify what's unique.

## Hypothesis

Block 0 may have a more concentrated activation pattern (lower rank covariance),
or handle a unique set of semantic patterns (embeddings → semantic space transform).
