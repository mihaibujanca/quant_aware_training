# Audit — Cycle 087

- **Item**: 96 — Mixed-corpus stacking evaluation — GPT-2 FFN
- **Cycle**: 087
- **Auditor verdict**:

```
status=PASS
old_outcome=inconclusive
new_outcome=inconclusive
reason=Implementation correct: matched 2048 tokens per condition, OWT atlas for channel selection in both, 3 seeds, proper stacking of 4 blocks. Verdict criteria applied faithfully: +7.5% Wiki improvement (needs >=10%), +1.8% OWT degradation (within 3% limit). The Wiki improvement is statistically significant (2-sigma separated) but below threshold. INCONCLUSIVE is the correct call.
guardrail_flags=G6_minor
tsv_path=autoresearch/logs/experiments.tsv
recommendations=The unpolished numbers tell a different story: +6.4% Wiki improvement but +5.0% OWT degradation (borderline refuted). Polish uses OWT-only data for both conditions, which disproportionately rescues condition B's OWT weakness (25.7% vs 23.4% polish gain). The "inconclusive" verdict hinges on the polished numbers — if the experiment had reported unpolished, OWT degradation would be 5.0% (borderline refuted). Consider: (a) running polish on mixed-corpus data for condition B to get a symmetric comparison, (b) noting that the single-block +19% (cycle 078) attenuating to +7.5% across 4 stacked blocks is expected — the lstsq refit optimizes each block independently while stacking introduces cross-block error accumulation, (c) closing this line — the effect is real but too small to justify the engineering complexity of mixed-corpus probes.
interesting=false
interesting_reason=The attenuation from +19% single-block to +7.5% stacked is the expected behavior when composing independent per-block optimizations. No novel insight.
```

## Guardrail Details

**G1 (Implementation)**: PASS. Token counts matched: A uses 2048 OWT tokens, B uses 512 tokens from each of 4 corpora (Shakespeare, WikiText-2, OWT, C4) = 2048 total. Channel selection uses OWT atlas for both conditions — fair comparison. Activations collected via forward hooks with `_apply_activation` matching GPT-2's GELU. Output projection refit via `_fit_output_projection` (lstsq). Stacking is correct: all 4 blocks [7,5,4,1] compressed simultaneously via `swap_compressed_ffns`.

**G2 (Reproducibility)**: PASS. Three seeds (42, 123, 7) with `_set_seed` setting both `random.seed` and `torch.manual_seed`. Hyperparameters fully recorded in JSON: compression=6, polish_steps=500, lr=1e-4, seq_len=128, eval_sequences=200. Eval corpus uses a fixed held-out seed (999) separate from training seeds.

**G3 (Results)**: PASS. Polished means match JSON: A OWT=55.24, Wiki=111.13; B OWT=56.23, Wiki=102.74. Wiki improvement = (111.13 - 102.74) / 111.13 = 7.55% (reported 7.5%). OWT degradation = (56.23 - 55.24) / 55.24 = 1.79% (reported 1.8%). Criteria: needs >=10% Wiki improvement for confirmed — not met. Needs >5% OWT degradation for refuted — not met. INCONCLUSIVE is correct. Standard deviations are reasonable: OWT std < 1 PPL point for both conditions polished; Wiki std ~1.2-1.8 PPL points.

**G4 (Surprising?)**: NOT SURPRISING. Single-block benefit (+19%, cycle 078) attenuating to +7.5% across 4 stacked blocks is the expected outcome. Each block's lstsq refit is optimized independently, but stacking introduces cross-block error propagation. The mixed-corpus refit's advantage (broader coverage of the input manifold) is diluted by: (a) the error compounding across 4 blocks, and (b) the polish step which partially reconverges both conditions toward OWT performance.

**G5 (Learning quality)**: MODERATE. The experiment establishes that mixed-corpus refit provides a real but modest generalization benefit (+7.5% Wiki, 2-sigma significant) that survives stacking, at a small OWT cost (+1.8%). Actionable takeaway: mixed-corpus refit is not worth the complexity for the stacking pipeline unless the target deployment requires multi-domain robustness. If so, the polish step should also use mixed-corpus data for a symmetric comparison.

**G6 (Method — polish masking)**: MINOR FLAG. Both conditions are polished with OWT-only data (line 331-332: `polish_tokens = owt_probe_corpus.probe_holdout`). This creates an asymmetric polish: condition A (OWT refit) is further refined on the same distribution, while condition B (mixed refit) is partially pulled back toward OWT. Evidence: unpolished OWT degradation = 5.0% (borderline refuted), polished = 1.8% (comfortably within limit). Polish benefits B more on OWT (25.7% vs 23.4% gain). The verdict criteria were defined on polished results, so INCONCLUSIVE remains correct, but the unpolished comparison is arguably a purer measure of the refit effect. This does not invalidate the experiment but is worth noting for interpretation.
