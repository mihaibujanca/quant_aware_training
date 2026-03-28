status=PASS
original_outcome=INCONCLUSIVE
audited_outcome=INCONCLUSIVE
guardrail_flags=none
interesting=false

## Phase 1: Verdict Check

INCONCLUSIVE is correct. The criteria map cleanly:

- Confirmed requires: top-20% >= 70% coverage in majority of blocks AND Spearman >= 0.35. Result: 0/12 blocks reach 70% (range [0.053, 0.257]). Pareto fails decisively. Spearman rho=0.573 is above 0.35 but p=0.051, and the Pareto gate fails regardless.
- Refuted requires: Gini < 0.2. Result: mean Gini=0.316, all 12/12 blocks above null (0.179). Cluster non-uniformity is real.
- Inconclusive: cluster structure exists (Gini > null) but is not Pareto-concentrated. This is exactly the "LLM distribution is flat" clause.

The outcome logic in code (lines 324-329) correctly implements these gates.

## G1: Implementation Correctness

**Hook point: CORRECT.** `mlp.c_fc.register_forward_hook` (line 89) captures pre-GELU activations z. This follows the lesson from cycle 069 — using the hook on the actual FFN submodule rather than reconstructing from block-level hidden states.

**Jacobian computation: CORRECT.** J(x) = W_out @ diag(GELU'(z)) @ W_in (line 116). GELU gradient computed via autograd with `approximate="tanh"` matching GPT-2's GELU variant. Resulting J is (768, 768), SVD yields 768 singular values. L2-normalization of SV profiles (lines 119-121) is appropriate — it makes profiles comparable across tokens with different activation scales.

**K-means: CORRECT.** K=50, n_init=5, deterministic per seed. Labels used for both Gini and coverage.

**Gini coefficient: CORRECT.** Standard formula (line 134). Applied to cluster sizes (token counts per cluster), not to coverage or variance quantities.

**Cumulative coverage: CORRECT but conservative.** Clusters sorted by SIZE descending (line 163), and between-cluster SS accumulated. Between-cluster SS = n_c * ||c_mean - global_mean||^2 per cluster — this is the ANOVA decomposition. Sorting by size rather than by SS contribution means a large cluster near the global mean gets priority over a small far-out cluster. This biases AGAINST Pareto detection — if we sorted by SS contribution, coverage would be higher. But the item description says "descending size order," so the implementation matches the design.

**Null baseline: CORRECT.** Random label assignment (line 249) with same K=50. This produces approximately uniform cluster sizes (Gini ~0.18 for K=50 with N=512), providing a sensible reference. Null Gini is constant across blocks (same random assignment) while null coverage varies (depends on profile structure).

**Spearman: CORRECT.** Correlates Gini (per block) with 1-damage_8x (per block). The 1-minus transform makes higher Gini correlate with lower damage (easier to compress), which is the hypothesized direction.

## G2: Reproducibility

PASS. Three seeds (42, 123, 7), 512 tokens each. Gini std < 50% of mean for 12/12 blocks (the stability check on line 291-294). Cross-seed top-20% coverage also stable: block 0 std=0.037 on mean=0.148 (CV=0.25), block 11 std not shown but per-seed values consistent. The non-Pareto result is not a single-seed artifact.

## G3: Results Interpretation

The agenda entry accurately summarizes the findings. Key numbers all check out:
- Gini range [0.287, 0.351], mean=0.316 — matches JSON `ginis_by_block`
- Null Gini=0.179 — matches JSON `null_ginis_by_block` (all identical by design)
- Top-20% coverage [0.053, 0.257] — matches JSON `top20_coverages_by_block` (min at block 2, max at block 11)
- Spearman rho=0.573, p=0.051 — matches JSON exactly

The interpretation is fair: non-uniform structure exists (Gini substantially above null) but it is not Pareto-concentrated. The Pareto hypothesis — that a few large "regions" dominate the function's behavior — does not hold for GPT-2 FFNs when regions are operationalized as K-means clusters on Jacobian SV profiles.

## G4: Counterintuitive Results

No counterintuitive results. The finding that GPT-2 FFN Jacobian structure is diffusely distributed across many clusters (rather than Pareto-concentrated) is consistent with prior findings: item 80 showed effective rank 434-510 (not extremely low-rank), and item 81 (pending rerun) tests whether this structure is global or token-specific. A diffuse cluster distribution is what you'd expect when the effective dimensionality of Jacobian variation is moderate.

## G5: Coverage Metric Limitation

The between-cluster SS metric measures how well cluster centroids explain total variance, accumulated in size order. An alternative — sorting by per-cluster SS contribution — would give higher coverage values and is arguably a fairer test of Pareto structure. However, the gap is large: the best block (11) only reaches 25.7% at top-20%, far from 70%. Even optimistic resorting would not close this gap.

## G6: Method Failure vs Genuine Finding

Genuine finding. The method is well-implemented, the null baselines are appropriate, and the result is stable across seeds. The cluster non-uniformity (Gini > null) is real signal; the lack of Pareto concentration is a substantive negative result about GPT-2 FFN structure, not a method artifact. K=50 is a reasonable choice (512 tokens / 50 clusters = ~10 tokens/cluster average), and the high K-means R-squared (~0.97 for block 0) shows the clustering captures the data structure well.

## Recommendations

1. INCONCLUSIVE is the correct and final verdict. No rerun needed.
2. The positive Spearman trend (rho=0.573, p=0.051) is suggestive but not actionable at N=12 blocks. Do not over-interpret.
3. The diffuse cluster structure result is informative for compression strategy: it suggests uniform channel allocation (rather than region-weighted allocation) may be near-optimal, consistent with cycle 073 findings.
