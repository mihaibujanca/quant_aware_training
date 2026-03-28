status=PASS
old_outcome=inconclusive
new_outcome=inconclusive
reason=Verdict is correct: 10/12 blocks have rank_99 <= 500 (low-rank confirmed) but Spearman(rank_99, P1_frob_std) = -0.497, below the +0.5 threshold required for confirmed. The negative sign means higher rank correlates with LOWER P1 variation, opposite to naive expectation. Block 1 outlier at 588.4 (near random baseline 592) is noteworthy.
guardrail_flags=none
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Item 81 (subspace alignment) is the natural next step to determine if the low-rank structure is globally shared. Block 1's near-full-rank status should be flagged as a special case — it likely has uniformly active GELU neurons and may behave differently under compression.
interesting=true
interesting_reason=Block 1 is a structural outlier with rank_99=588 (near random 768x768 baseline of 592), suggesting nearly inactive GELU gating. Rank does NOT correlate with compression damage (rho=-0.33, p=0.30), meaning low rank is necessary but not sufficient to explain compressibility.
