status=PASS
old_outcome=confirmed
new_outcome=confirmed
reason=Recovery 73.8% (3 seeds, std ~1.1%) well above 30% threshold. Per-seed recoveries 72.9-75.2% show excellent consistency. Implementation is correct (low-rank additive correction on residual stream, zero-init U). Note: absolute corrected PPL (70.1) is substantially worse than k=4 corrected (59.1) despite matching recovery %; the "graceful scaling" claim holds for % recovery but not absolute quality.
guardrail_flags=minor: uncorrected PPL sanity bounds (L532) reference cycle 106 polish baseline (~93.9) which is not comparable to k=5 stacked uncorrected; bounds should be recalibrated
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Test k=8 cascade corrections (original intent of item 133 before timeout). Consider reporting absolute residual alongside % recovery to avoid overstating k=5 quality.
interesting=true
interesting_reason=Recovery % is remarkably stable (74.3% at k=4 vs 73.8% at k=5) despite 70% larger degradation, suggesting correction capacity scales proportionally with damage — the low-rank correction is capturing a fixed fraction of the error structure regardless of how many blocks are compressed.
