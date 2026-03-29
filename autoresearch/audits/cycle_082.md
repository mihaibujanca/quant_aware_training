status=PASS
old_outcome=inconclusive
new_outcome=inconclusive
reason=Verdict INCONCLUSIVE is correct. True GN via jvp+vjp implemented correctly (CG residuals monotonically decrease, confirming PSD J^T J). GN +21.5% is indistinguishable from steepest descent +21.6% because the system is severely underdetermined (1016 tokens << 98K params, rank(J^T J) <= 1016). The curvature information is effectively wasted. This is not a bug — it is the expected behavior of a single Newton step in the null-space-dominated regime.
guardrail_flags=G4-addressed: GN matching steepest descent is expected (not a bug) given rank-deficient J^T J. Stale comment on line 13 says HV_SEQS=64 but actual config is HV_SEQS=8.
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Close the GN line for single-step correction on GPT-2 FFN — even 64 seqs (8K tokens) is still 12x underdetermined. If revisiting, try: (1) iterative Newton-CG (multiple steps like Adam but with curvature), (2) block-diagonal approximation reducing effective param count, or (3) subspace-restricted Newton in the top-k gradient PCA directions.
interesting=true
interesting_reason=Confirms that single-step second-order methods are fundamentally limited by data-to-parameter ratio in this compression polishing setting. Future polishing experiments should prefer iterative first-order methods (Adam) or reduce the effective parameter space before applying Newton.
