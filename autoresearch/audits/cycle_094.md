# Audit — Cycle 094

- **Item**: 110 — Non-uniform compression allocation — GPT-2 FFN
- **Cycle**: 094
- **Auditor verdict**:

```
status=PASS
old_outcome=refuted
new_outcome=refuted
reason=Uniform PPL=69.40, mild=69.56 (+0.24%), aggressive=70.25 (+1.2%). Non-uniform is equal or slightly WORSE than uniform. Refutation threshold was <3% improvement; the best non-uniform allocation achieves 0% improvement. Budget sums verified (all 2048). 3 seeds, per-block polish fair across allocations. Result is predicted by cycle 093 (c ratio-independent): if the cascade amplification factor is a network property independent of per-block compression quality, then redistributing channels cannot help — the bottleneck is inter-block error propagation, not per-block approximation.
guardrail_flags=none
tsv_path=autoresearch/logs/experiments.tsv
recommendations=The allocation line of inquiry is now exhausted (ratio-independent c + allocation-independent stacking). Next experiments should target the cascade mechanism itself: (1) block-wise error feedback (correct block k before compressing block k+1), (2) joint multi-block polish (optimize all W_down simultaneously to capture cross-block gradient interactions).
interesting=false
interesting_reason=Result is fully predicted by cycle 093. No new information about the stacking problem.
```

## Guardrail Details

**G1 (Implementation)**: PASS. Three allocations all sum to TOTAL_BUDGET=2048 (assert at line 75-77). Channel counts map correctly: uniform gives 512 each, mild shifts 102 channels from block 7 to block 1, aggressive shifts more aggressively. The `per_layer_hidden` field in `CompressionBudgetSpec` correctly overrides the default `hidden_fraction`. Per-block polish (100 steps Adam, lr=1e-4) is identical across allocations. Polish is done on single-block students, then weights are injected into the full 4-block student for eval. This is fair since all allocations follow the same procedure.

**G2 (Seeds)**: PASS. 3 seeds (42, 123, 7). Per-seed values stored in JSON. Seed 42 consistently produces higher PPL across all allocations (80.4-81.4), seed 123 lowest (62.1-62.7). The allocation ranking (uniform best, aggressive worst) is consistent across all 3 seeds, confirming it is not a noise artifact.

**G3 (Results)**: PASS. The numbers match between manifest, JSON, and agenda. Verdict logic is correct: best non-uniform delta is +0.24% (worse, not better), which is > -3%, triggering REFUTED. The thresholds (CONFIRMED at <= -10%, REFUTED at > -3%) are reasonable for this effect size. One could argue aggressive being +1.2% worse hints that spreading channels too thin actively hurts, but this is within noise (std ~9.7).

**G4 (Surprising?)**: Not surprising. Cycle 093 showed c is ratio-independent (~0.41 regardless of 3x/4x/6x). If the per-block damage rate through the cascade doesn't depend on channel count, reallocating channels between blocks should be irrelevant. This experiment confirms that prediction. No counterintuitive-result investigation needed.

**G5 (Learning quality)**: PASS. Baseline PPL = 51.66 +/- 5.49, within the [40, 65] abort range. Uniform 6x at 4 blocks gives +34.3%, which is lower than the +56% from cycle 089's k=4 stacking — but cycle 089 stacked 4 consecutive blocks from the damage ordering (blocks 7,5,4,1), and the delta difference is consistent with seed variance and the slightly different eval split sizes. Not a concern.

**G6 (Method)**: No failure. The experiment directly tests the hypothesis with a fair comparison. Minor design note: polishing blocks independently means cross-block gradient interactions are missed, but this applies equally to all allocations, so it does not bias the comparison. A joint-polish variant could be a future experiment but would test a different hypothesis.
