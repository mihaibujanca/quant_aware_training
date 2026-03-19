# Audit — Cycle 054

**Item**: 70 — Training-distribution probing with OpenWebText subset — GPT-2 FFN
**Verdict under review**: REFUTED (+48.21% corrected PPL degradation, threshold +30%)

## G1: Implementation Correctness — PASS

- OpenWebText correctly loaded via `load_dataset("Skylion007/openwebtext")` (standard HF dataset).
- Used as BOTH probe (geometry + correction fitting) and eval corpus — matching the item's intent.
- Compression pipeline: k=4 blocks [7,5,4,1] at 6× (`HIDDEN_FRACTION=1/6`), topk selection. Matches prior cycles.
- Correction: lstsq affine fit at layer 8 on `probe_train`, evaluated on `eval_val` — proper train/eval split, no data leakage.
- 80/20 document split via `_read_text_chunks`, seeded shuffle — correct.

## G2: Reproducibility — PASS

- 3 seeds: [42, 123, 7]. Per-seed corrected: +54.01%, +45.68%, +44.94%.
- Mean: +48.21% ±5.03 (std). All 3 seeds individually exceed the +30% refutation threshold.
- Variance (std=5.03) is higher than WikiText-2 (std=1.02), likely reflecting OWT's greater content diversity across 200-sequence subsamples.

## G3: Results Interpretation — PASS WITH CAVEAT

The REFUTED verdict is correct — all 3 seeds exceed +30%.

**Caveat**: The manifest claims OWT (+48%) performs "comparable to WikiText-2 (+44%)". This is misleading when stated without context. OWT has LOWER baseline PPL (~52 vs ~70 for WT2) because GPT-2 was trained on WebText. The same absolute PPL increase produces a larger relative percentage when the denominator is smaller. In absolute corrected PPL, OWT (~77) is actually BETTER than WT2 (~101). The relative comparison overstates the gap. That said, this does not change the verdict — +48% degradation is still catastrophic.

## G4: Counterintuitive Result Check — EXPLAINED

OWT (+48%) being worse than WT2 (+44%) in relative terms is NOT a bug. It is a denominator effect: OWT baseline is ~52 PPL, WT2 baseline is ~70 PPL. A 25 PPL absolute increase is +48% on base 52 but only +36% on base 70. The comparison is technically valid (relative degradation measures user-perceived quality loss) but should be interpreted cautiously.

Baseline PPL sanity: GPT-2 on OWT at seq_len=128 yielding ~52 PPL is plausible (published ~30 PPL is at context 1024; shorter context = higher PPL). Consistent with WT2 baselines (~70) in the same framework.

## G5: Method Failure vs Genuine Refutation — GENUINE

Consistent evidence chain:
- Cycle 044: WT2 corpus-specific compression = +31.86%
- Cycle 045: WT2 with full recipe = +44.31%
- Cycle 054: OWT training-distribution probe = +48.21%

The fundamental issue is that k=4@6× compression destroys too much information for broad-distribution text. Shakespeare works (+8.5%) because its narrow distribution uses few FFN channels. Training-distribution probing cannot rescue the method because the problem is the compression ratio, not the probe mismatch.

## Summary

| Guardrail | Status |
|-----------|--------|
| G1: Implementation | PASS |
| G2: Reproducibility | PASS (3 seeds) |
| G3: Interpretation | PASS with caveat (denominator effect) |
| G4: Counterintuitive | EXPLAINED (not a bug) |
| G5: Genuine refutation | YES |

**Audit status**: ACCEPT — verdict REFUTED is correct.

**Recommendation**: Note in the agenda that the OWT-vs-WT2 relative comparison is affected by baseline PPL differences. Future corpus comparisons should report both absolute and relative PPL to avoid confusion.
