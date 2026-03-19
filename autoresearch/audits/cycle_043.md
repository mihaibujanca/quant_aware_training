status: PASS
original_outcome: inconclusive
audited_outcome: inconclusive
guardrail_flags: G2 (single seed)
reason: INCONCLUSIVE is correct. k=4 multi-point is worse than single-point (corrections interfere), k=5 multi-point at +12.08% is within rounding of the 12% threshold but not a clear win. The per-correction MSE breakdown is informative: block 1 correction reduces MSE by 40.8% but block 4's correction increases MSE by 41.3% — the cascading corrections distort the residual stream for downstream blocks. Single-point correction at the end remains the optimal no-SGD strategy.
recommendations: The correction interference reveals that the error is non-separable across blocks — each correction changes the input distribution for subsequent blocks. Future work: jointly optimize all corrections (but that requires iterative fitting, approaching SGD). For now, single end-point correction is the practical answer.
