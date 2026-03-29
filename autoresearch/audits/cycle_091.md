# Audit — Cycle 091

- **Item**: 86 — Mutual information bottleneck channel selection — GPT-2 FFN [MOONSHOT]
- **Cycle**: 091
- **Auditor verdict**:

```
status=VERDICT_CORRECTED
old_outcome=refuted
new_outcome=inconclusive
reason=Experiment is well-implemented and results are reproducible across 3 seeds. MI-greedy selects substantially different channels from topk (35-61% overlap), but PPL difference is <1% in all blocks. The agenda criteria explicitly state "Inconclusive if: MI-greedy selects different channels but PPL improvement is < 3%." This is exactly what happened. The code's verdict logic (line 509) maps this to REFUTED via a fallthrough, but the written criteria map it to INCONCLUSIVE. Correcting verdict accordingly.
guardrail_flags=G3
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Verdict in agenda.md and JSON should be updated to INCONCLUSIVE. The result is practically negative (MI adds nothing over topk) but the criteria distinction matters for the research log. No follow-up experiments warranted — the topk heuristic is sufficient for channel selection at 6x compression.
interesting=false
interesting_reason=Null result: an expensive selection method (O(k*D) Schur updates) finds different channels but identical PPL. Confirms topk heuristic sufficiency, which was already the working assumption. No new structure discovered.
```

## Guardrail Details

**G1 (Implementation)**: PASS. MI greedy forward selection correctly minimizes det(Sigma_{y|S}) via Schur complement rank-1 updates. Float64 used for numerical stability. Full Schur recomputation every 50 steps guards against accumulated rank-1 update drift. Cholesky with fallback to lstsq/eigvalsh. Covariance estimated from separate corpus (seed+1000 offset) to avoid data reuse with lstsq fitting. Regularization (1e-6) added to all PD matrices. The selection criterion (maximize g^T S^{-1} g / sigma_{c|S}) is the standard Gaussian conditional MI gain.

**G2 (Reproducibility)**: PASS. 3 seeds (42, 123, 7), 4 blocks each. All hyperparameters explicit: k=512, ratio=6, COV_SEQUENCES=64 (8192 tokens), LSTSQ_SEQUENCES=32 (4096 tokens), EVAL_SEQUENCES=200, MI_RECOMPUTE_EVERY=50, REG_LAMBDA=1e-6. Per-seed raw PPL values stored in JSON. Results consistent across seeds (no seed-dependent outliers).

**G3 (Results)**: FLAGGED — VERDICT MISMATCH. Agenda criteria:
- Confirmed (MI >=5% better in >=2/4 blocks): NOT MET (max 0.64%)
- Refuted (>=90% channel overlap): NOT MET (max 61%)
- Inconclusive (different channels, <3% improvement): MET

The experiment code (line 509) treats the case `blocks_mi_better_3pct < 2` as REFUTED, but the agenda text maps "different channels + <3% improvement" to INCONCLUSIVE. The manifest notes this discrepancy ("TECHNICALLY THIS — but refuted on practical grounds") but applied REFUTED anyway. Per the written criteria, the correct verdict is INCONCLUSIVE.

Numerical spot-checks: Block 1 seed 42: topk_ppl=122.42, baseline=123.36 -> rel=-0.76%. JSON says -0.756%. OK. Block 11 seed 7: topk_ppl=128.65, baseline=124.99 -> rel=+2.93%. JSON says +2.928%. OK. Overlap block 1 seed 42: 0.363 (186/512). JSON matches.

**G4 (Surprising?)**: NOT FLAGGED. The result that different channels yield similar PPL is consistent with known high redundancy in GPT-2 FFN hidden dimensions. Multiple subsets of 512/3072 channels can reconstruct the output equally well via lstsq.

**G5 (Learning quality)**: ADEQUATE. Random baseline 6-7% worse than both structured methods confirms both capture real signal. The experiment cleanly answers the question (does MI-optimal selection improve over topk?) with a clear negative. However, N/D=2.67 for covariance estimation is thin — the Gaussian MI estimate may be imprecise, though this wouldn't change the conclusion since the random baseline confirms both methods work.

**G6 (Method)**: PASS. The Gaussian MI assumption (post-GELU activations are not Gaussian) is a known limitation but standard in the column subset selection literature. The greedy forward selection is the correct algorithm (backward elimination from 3072 would be intractable). Using separate corpora for covariance and lstsq is good practice. Single-block swap evaluation isolates the channel selection effect cleanly.
