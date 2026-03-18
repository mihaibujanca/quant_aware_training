# Audit — Cycle 009

## Item
19 — Jacobian variance atlas

## Checklist

1. **Criteria match**: The agenda says "Refuted if: all blocks have L < 0.6." All 12 blocks have L < 0.6 (max is 0.382). Refuted criterion is met cleanly.

2. **Methodology**: Linearity index L = 1 − mean(||J(x) − J̄||_F / ||J̄||_F) computed per block with 256 probe tokens. The formula is correct per the agenda description. J(x) is computed via W_out @ diag(σ'(z)) @ W_in which is the standard GELU FFN Jacobian.

3. **Negative L values**: L < 0 occurs when mean_ratio > 1, meaning the average per-token deviation from the mean Jacobian exceeds the mean Jacobian norm. This is mathematically valid and indicates high Jacobian diversity. Not a bug.

4. **Correlation**: Spearman ρ = −0.14 with p = 0.66 on 12 data points. Non-significant and near zero. L does not predict compression difficulty.

5. **Proxy label**: Correctly labeled as proxy (gpt2 + Shakespeare).

6. **Downstream impact**: Item 21 (single-affine collapse on GPT-2) depends on item 20 (toy), not on item 19 directly. But this result makes item 21 less promising — if no block is remotely near-linear, single-affine collapse will fail. Item 20 (toy) remains worth testing as a methodological baseline.

## Status: PASS

The refuted outcome is correct and well-supported. No correction needed.
