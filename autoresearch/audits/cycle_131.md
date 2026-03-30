# Audit — Cycle 131, Item 88: Learned Bottleneck AE for FFN Compression POC

**status=PASS**

**old_outcome=inconclusive**
**new_outcome=inconclusive**

## Verdict Justification

AE achieves 2.0–3.4% PPL improvement over topk+lstsq at k=384 (55.76–55.95 vs 57.23–57.83), consistent across 3 seeds with low variance (σ ~0.3–0.4). Nonlinearity is 4.6–6.3%, clustering at ~5% — right at the INCONCLUSIVE boundary (neither CONFIRMED ≥5% nor REFUTED <2%). This result does not cross the 5% improvement threshold, placing it squarely in the indeterminate zone.

## Critical Caveat: Param Mismatch

**AE has 2.5x more parameters than topk.** topk is a pure selection mechanism (near-zero params), while the AE adds 1.475M params per block. This is a **major confound** — the performance gap cannot be attributed to learned geometry alone; it conflates model capacity with compression quality. Any follow-up must either:
1. Reduce AE to match topk's param count (e.g., reduce bottleneck or hidden layer width), or
2. Explicitly test param-matched variants and report the tradeoff.

## Positive Signals (Not Decisive)

- Both AE and topk beat the float baseline (57.65), indicating that the FFN's original capacity is partly redundant for the task.
- The AE's model-loss training (auxiliary loss on reconstructing pre-ReLU activations) captures task-relevant structure that pure compression + lstsq correction misses.
- Consistency across blocks 4 and 7 and seeds suggests a real effect, not noise.

## Guardrail Flags

- **G2_param_mismatch**: The 2.5x param gap invalidates any claim of efficiency. Follow-ups must isolate the contribution of learned parameters vs. nonlinearity.

## Recommendations for Follow-Up

1. **Param-matched test**: Reduce AE bottleneck to ~150–200 dims (matching topk's effective capacity) and remeasure PPL gap.
2. **Aggressive compression**: Test at k=128 or k=256 where topk+lstsq degrades further (expect larger AE margin, better test of the method).
3. **Stacking**: Chain 2–3 AE-compressed blocks and measure cascade error — does the advantage compound or cancel?

## Session Notes

- Manifest is well-written with clear learned/next sections.
- JSON structure is clean; all 6 runs present with full hyperparameters and nonlinearity metrics.
- Verdict assignment (INCONCLUSIVE) is correct.

**Audit result: PASS** (verdict matches evidence; caveat clearly flagged for interpretation.)
