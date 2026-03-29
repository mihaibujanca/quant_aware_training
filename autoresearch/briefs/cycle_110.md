# Cycle 110 Brief — Channel-Level Bump Anatomy: Compression Relevance (GPT-2 FFN)

**Item:** 121 | **Type:** analysis (interpretability) | **Depends on:** none

## Context
Cycle 103 found 94.6% of GPT-2 FFN activations fall in the GELU bump zone (|z|<2).
Cycle 108 showed early blocks have tight concentration in the GELU knee (σ<0.5).
The top-20 bump-concentrated channels per block have 99%+ activations in the zone.
Open question: do these bump-dominated channels overlap with compression-important channels?

## Question
Is there a positive Spearman correlation between per-channel bump fraction and
per-channel importance across GPT-2's 12 FFN blocks?

## Key concern
The existing importance formula (from `preactivation_anatomy_gpt2.py`) includes
`mean(|GELU'(z_j)|)` and `std(GELU(z_j))` — both depend on z_j's position relative
to the bump zone, creating partial circularity with bump_fraction. Cycle 108 review
(A2) flagged this. Plan must decompose importance into tautological vs non-tautological
components.

## Assets
- `experiments/preactivation_anatomy_gpt2.py` — importance scoring, pre-activation hooks
- Cycle 103/108 results in `autoresearch/results/`
