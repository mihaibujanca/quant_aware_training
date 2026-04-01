# Cycle 175 Audit — Item 186

## Verdict: CONFIRMED (with caveats)

Mean KL > 0.1 in 9/12 blocks. The "semantic pattern" flag triggers on repeated tokens
in top-50 across blocks, but these are mostly common tokens (-, the, and, commas, BPE
fragments). This is token frequency overlap, not a genuine semantic pattern.

Strictly by the item criteria: CONFIRMED (KL threshold met, automated pattern check passes).
In honest interpretation: the semantic pattern criterion is weakly met at best.

## Key metrics
| Block | mean_KL | frac_relu_better | ||y_neg|| | ||s_neg|| |
|-------|---------|------------------|-----------|-----------|
| 0     | 1.419   | 0.180            | 16.1      | 6.0       |
| 1     | 0.079   | 0.387            | 13.1      | 5.9       |
| 2     | 0.173   | 0.357            | 27.6      | 6.9       |
| 3     | 0.245   | 0.324            | 34.3      | 7.1       |
| 4     | 0.283   | 0.314            | 34.2      | 7.1       |
| 5     | 0.254   | 0.293            | 33.8      | 7.9       |
| 6     | 0.190   | 0.314            | 30.6      | 8.7       |
| 7     | 0.109   | 0.383            | 30.1      | 9.7       |
| 8     | 0.107   | 0.400            | 29.6      | 11.2      |
| 9     | 0.067   | 0.519            | 31.3      | 13.1      |
| 10    | 0.054   | 0.547            | 54.9      | 15.9      |
| 11    | 0.257   | 0.316            | 124.7     | 18.7      |

Correlations: r(KL, ||y_neg||) = -0.16, r(KL, ||s_neg||) = -0.33

## Interpretation

1. **GELU→ReLU changes predictions substantially** (9/12 blocks > 0.1 nats), but the
   impact is NOT driven by the negative regime. Correlations with ||y_neg|| and ||s_neg||
   are weak and negative.

2. **Block 0 dominates** (KL=1.42, frac_relu_better=0.18). Early layers are most
   sensitive to the GELU/ReLU difference, likely because GELU's smooth gating near z=0
   matters most when the residual stream is least redundant.

3. **Middle blocks (7-10) have low KL and ~50% frac_relu_better** — the modification
   is nearly neutral there, consistent with cycles 173-174.

4. **Block 11 has elevated KL (0.26)** despite being the final block — the last-mile
   nonlinearity shape matters for token prediction.

5. **ReLU hurts predictions 60-80% of tokens in blocks 0-6** (frac_relu_better < 0.4),
   but is roughly coin-flip in blocks 9-10 (0.52-0.55). This means the GELU shape
   genuinely matters for the positive regime (z near 0) in early/middle blocks.

## Consistency with prior cycles
- Cycles 173-174 said the negative regime is noise/absorbable as bias. This cycle
  confirms that: correlations with negative norms are near zero.
- BUT the GELU→ReLU swap is NOT just a negative-regime intervention. It also changes
  the positive regime (GELU(z) ≠ z for z near 0). The large KL values are driven by
  this positive-regime shape difference, not the negative regime.

## Methodological note
The item specification says "ReLU-at-block-l model" which is what we implemented.
However, this conflates two effects: (a) zeroing the negative GELU outputs, and
(b) changing the positive-regime activation shape. A cleaner test of negative-regime
impact would use max(0, GELU(z)) (zero the negative, keep GELU shape for positive).
That's what cycle 173 effectively tested, and it found negligible impact.

## Code review
- Hook correctly captures c_fc output and replaces GELU with ReLU in c_proj
- Sanity check confirms P_relu ≠ P_float (max logit diff > 0 for all blocks)
- KL computed correctly as sum(P_float * (log_P_float - log_P_relu))
- No bugs found
