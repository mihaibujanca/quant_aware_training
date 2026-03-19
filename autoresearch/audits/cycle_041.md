status: PASS
original_outcome: confirmed
audited_outcome: confirmed
guardrail_flags: none
reason: 5/6 blocks clearly pass the ≤10% threshold (all within ±1.5%). Block 0 is a consistent catastrophic failure (+29.74%), matching GPT-2-small's block-0 pattern. The result is clean and multi-seed (3 seeds, low std 0.20-1.30%). GPT-2-medium baseline PPL (94-97) is well-calibrated. The method generalizes without any code changes.
recommendations: The block-0 failure pattern is universal across model sizes — investigate what makes first-layer FFN special (likely: the embedding→hidden transition concentrates information in all channels). For the compression summary, GPT-2-medium enables ~20/24 = 83% of blocks as candidates (excluding blocks 0, 23, and boundary blocks).
