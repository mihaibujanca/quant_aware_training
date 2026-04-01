# Cycle 199 Review: K-sweep for near-rank-1-input blocks (Item 209)

## Verdict: REFUTED

Near-rank-1 input (blocks 4,7,11 from cycle 198) does NOT enable smaller K.
PPL degrades monotonically from +57% at K=8 to +26% at K=512 with no plateau.

## Key Numbers

| K   | PPL   | vs float |
|-----|-------|----------|
| 8   | 84.81 | +57.4%   |
| 16  | 83.97 | +55.8%   |
| 32  | 79.47 | +47.5%   |
| 64  | 76.94 | +42.7%   |
| 128 | 74.77 | +38.7%   |
| 256 | 71.85 | +33.3%   |
| 512 | 68.14 | +26.4%   |

Float baseline: 53.89 PPL (on held-out eval split).

Note: Absolute PPL is higher than cycle 196 (K=512 here: 68.14 vs 30.09 in cycle 196)
because this run uses a SEPARATE held-out eval split (sequences 200-300 of the validation set),
while cycle 196 measured PPL on the same 200 sequences used for fitting (optimistic/overfit
measurement). The degradation RATIOS are comparable: +26% here vs +20% in cycle 196.

## Interpretation

If rank-1 input were the binding constraint, we would expect PPL to plateau at K~4-16
(where the rank-1 manifold is fully covered). Instead, PPL improves by ~1.5-2% per doubling
of K across the full range tested. This is consistent with the FFN performing operations that
span many input directions (including token-specific high-frequency features from the LN output),
not just the dominant PCA direction.

The rank-1 finding (cycle 198) reflects the input to the BLOCK, not the input to the MLP.
After LayerNorm, the per-token centering and scaling creates token-specific variation that the
MLP must capture across all 768 LN-output dimensions, regardless of the residual stream's
low effective rank. Block input rank ≠ MLP input diversity.

## Impact

Closes item 209. The rank-1 residual stream is a property of the residual DIRECTION, not of
the full LN-normalized token vectors. K reduction based on input rank is not viable.
Cascade barrier and compression floor remain at K=512 for quality ~+26% above float PPL.
