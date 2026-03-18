status: PASS
original_outcome: confirmed (proxy)
audited_outcome: confirmed (proxy)
guardrail_flags: G2 (single random seed for random ordering)
reason: Easiest-first clearly outperforms hardest-first (27-36% perplexity reduction at k=6-9), exceeding the ≥25% threshold. The gap with random is small (1-2%) because random orderings don't systematically pick worst layers — this is expected and does not undermine the finding. Single-block localization is informative as a signal of which blocks to avoid compressing early. Methodology is sound: teacher ppl verified, k=12 converges identically for all orderings (sanity check), results are internally consistent.
recommendations: (1) Run multiple random seeds to confirm random ordering typically falls between easiest and hardest. (2) Test whether the easiest-first advantage holds at 8x compression (less aggressive). (3) Investigate whether partial compression (k=4-5 easiest blocks only) gives a viable operating point for practical use. (4) Proceed to items 19-20 for geometric analysis of why specific blocks are easy/hard.
