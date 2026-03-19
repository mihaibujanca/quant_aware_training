verdict: APPROVE
round: 1
issues: [minor — Jaccard definition and null baseline]
feedback:

**R1 (Premise validity):** Sound. Channel selection stability under varying probe corpus size is a well-motivated practical question. The hypothesis that centroid statistics converge with small data is plausible — activation magnitude statistics often stabilize quickly — and the experiment would produce actionable deployment guidance either way.

**R2 (Measurement soundness):** Minor note. The plan computes Jaccard as 3-way (intersection/union across all 3 seeds), but the P5 null baseline of ~0.17 is the pairwise random Jaccard. The 3-way random Jaccard for k=512 from 3072 is ~0.001 (much lower). This makes the 0.85 threshold extremely conservative relative to null — which is fine, the threshold is safe. However, note that 3-way Jaccard is systematically lower than pairwise: if pairwise Jaccard is 0.90, 3-way could be ~0.82 due to independent noise compounding. Consider reporting both pairwise mean and 3-way Jaccard so the 0.85 threshold can be interpreted correctly. This is a documentation/reporting suggestion, not a validity issue.

**R3 (Seeds/statistics):** 3 seeds specified, 18 total runs. Adequate. PPL eval on fixed 200-seq holdout is sufficient for relative comparisons. The validation check (Jaccard@2048 > 0.9) is a good self-consistency control.

**R4 (Method-architecture alignment):** Correct. The atlas builder hooks at `module.c_fc(x)` (FFN input, post-LN), which is the standard measurement point. GELU compatibility is noted. The `max_tokens_per_layer` parameter correctly caps probe tokens at the atlas level, and `probe_holdout_sequences = ceil(probe_size/128)` ensures enough sequences are available. API usage matches the codebase.

**R5 (Pitfall scan):**
- MP1 (threshold above trivial): Not applicable — 0.85 is well above the null (~0.001 for 3-way, ~0.17 for pairwise).
- MP2 (detection mismatch): Not applicable — centroid statistics are activation-agnostic.
- MP3 (underdetermined fitting): Not applicable — no model fitting, only channel selection.
- MP4 (domain mismatch): Not applicable — testing on the same domain (GPT-2 FFN).
- MP5 (capacity): Not applicable — GPT-2 is pretrained, single-block swap is the evaluation.
- MP6 (verdict misclassification): Criteria are clear and measurable. Low risk.
- MP7 (post-LayerNorm): The atlas hooks post-LN, but this is the standard FFN compression measurement point used in all prior cycles. Not a concern.
- MP8 (representation vs functional rank): Not applicable — measuring channel selection, not intrinsic dimension.

**Overall:** Clean plan with good controls (self-consistency check at 2048, fixed eval corpus, channel count assertion, PPL sanity check against known values). The fallbacks are sensible. No issues that would compromise validity.

Minor suggestion: report both pairwise-mean and 3-way Jaccard in the output JSON for completeness. The 3-way metric is a stricter test and could lead to premature refutation at the margin if the pairwise scores are solid but the 3-way dips below 0.85 due to one outlier seed.
