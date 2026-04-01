# Cycle 185 Review — Compression Ratio Sweep k={2,4,6,8} (Item 194)

Item 194. Reviewer: design-review agent.

## R1: Premise validity — OK

The recipe has been validated at k=4 (cycle 183: 59.4 PPL, 6x compression). Sweeping k is the natural next step to understand the PPL-vs-compression tradeoff curve. No novel algorithmic claims — purely empirical characterization.

## R2: Measurement soundness — OK

Single metric (stacked PPL) is the right choice for a compression sweep. Teacher baseline PPL sanity check ([15, 65]) guards against data loading bugs. k=4 cross-check against cycle 183 (~59.4) provides regression detection. Compression ratios are computed from actual param counts, not theoretical.

## R3: Capacity and statistical adequacy — OK

3 seeds (42, 137, 7) per k-value. 12 total runs. This is adequate for a parameter sweep where the independent variable (k) spans 4x range. Cycle 183 showed std ~8 PPL at k=4 across 3 seeds, so the sweep should resolve a monotonic trend if the effect size is >8 PPL between adjacent k-values (which is expected given 2x parameter change at each step).

## R4: Method-architecture alignment — OK

The plan reuses exactly the same topk+lstsq+polish pipeline from cycle 183 with only STUDENT_HIDDEN changed. No new hooks, no new training dynamics. The k=8 case (hidden=1024 > GPT2_HIDDEN=768) means the student MLP's hidden layer is larger than the model's residual stream — this is fine architecturally (768->1024->768 is a valid MLP) but note that at k=8 the student retains 1024/3072 = 33% of channels, which is relatively mild compression. The lstsq fitting and polish should work identically at all k values.

## R5: Known pitfall scan — OK

- **MP3 (Underdetermined fitting)**: At k=8, output projection has 1024*768 + 768 = 786K params, fitted from MAX_TOKENS_PER_LAYER=4096 tokens x 1024 hidden dims. The lstsq system is (4096, 1024) -> (4096, 768), which is overdetermined (4096 > 1024). OK.
- **MP5 (Insufficient capacity)**: Each k value has different capacity by design. k=2 (256 hidden) may underfit badly — that's informative, not a flaw. The question is whether k=8 has enough capacity to close the gap to float. No issue.
- **MP6 (Verdict misclassification)**: The criteria are well-defined and the metric (PPL) is the standard task metric. No risk of uninformative measurement.
- **MP7 (Post-LayerNorm)**: Not applicable — no geometric measurements, just PPL.
- **MP8 (Representation vs functional rank)**: Not applicable — this is a compression sweep, not a rank analysis.
- No MP1, MP2, MP4 issues.

## Summary

No issues found. This is a clean parameter sweep with well-defined criteria, adequate seeds, and a tested pipeline. The only subtlety is that k=8 retains 33% of FFN channels (mild compression) — if it still shows high PPL, that's a strong signal about the compression floor.

**Verdict: ACCEPT.**
