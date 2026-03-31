status=PASS
old_outcome=refuted
new_outcome=refuted
reason=AE stacking (74.4) clearly worse than topk (71.0). Pre-polish AE stacking catastrophic (259.8). Single-seed but result is unambiguous (4.7% gap, much larger than typical seed noise of ~0.5).
guardrail_flags=G2_single_seed
recommendations=The composition problem is the fundamental barrier. Next: test JOINTLY trained AE (train all 4 blocks under stacked model loss simultaneously, not independently). This is the only untested approach that could address the cascade.
interesting=true
interesting_reason=The pre-polish stacking PPL (259.8 for AE vs 84.9 for topk) reveals a fundamental asymmetry: topk preserves approximate compositionality by construction (it keeps actual GELU channels), while AE learns a completely different representation that doesn't compose. This is the strongest evidence that composition-awareness must be built into the compression method, not added post-hoc.
