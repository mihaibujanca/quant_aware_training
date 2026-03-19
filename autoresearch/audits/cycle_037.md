status: PASS
original_outcome: refuted
audited_outcome: refuted
guardrail_flags: low-seed-count
reason: k=4@6x achieves 27.8% FFN param reduction at +9.77% PPL (2 seeds, std=0.26%), meeting the >=25% reduction and <10% PPL criteria. All six k=5 heterogeneous configs exceed +15% PPL (best: k5_block6_4x at +16.99%), confirming a hard block-count wall — even 2x compression on the 5th block gives +20%. Param counting logic verified: source across all 12 blocks, compressed + uncompressed correctly summed. The refutation is genuine (method works at k=4; the compounding through the residual stream is the binding constraint at k=5, not a method failure). Verdict logic in code matches the pre-registered criteria.
recommendations: 2 seeds is thin; if k=4@6x becomes a load-bearing result for downstream work, add 2-3 more seeds. The positive k=4 finding is correctly noted in the agenda entry and should inform future budget allocation experiments.
