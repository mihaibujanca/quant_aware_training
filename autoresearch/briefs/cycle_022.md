# Cycle 022 Brief — Attention Head Geometry Census (GPT-2)

## Item
40: Attention head geometry census — GPT-2

## Question
Is there geometric compression opportunity in attention, not just FFN?

## Approach
For each of 144 heads (12 layers × 12 heads):
1. Effective rank = nuclear_norm(A_mean) / spectral_norm(A_mean) where A_mean is the mean
   softmax attention matrix over probe sequences.
2. Head output norm variation = std of ||head_output||_2 across tokens (low = more compressible).
3. Pairwise CKA between heads in same layer on head output matrices (N_tokens × d_head).

Prunability score = (1 - eff_rank_norm) × (1 - variation_norm) × redundancy

Intervention: zero out top-K heads via zeroing their c_proj rows (K = 5, 10, 20, 30).
Measure perplexity change.

## Criteria
- Confirmed: ≥20% heads have eff_rank < 0.3×64=19.2 AND zeroing 10 most prunable ≤5% ppl increase
- Refuted: all heads high rank AND zeroing 5 causes >15% ppl increase
- Inconclusive: low-rank heads are still functionally critical (low rank, high zeroing cost)

## Dependencies
- Item 9 (GPT-2 probe corpus setup) — same pattern as cycles 005–009
