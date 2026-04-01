# Audit — Cycle 194 (Item 204)

**Verdict:** REFUTED
**max_decrease=2/12 layers** (error magnitude decreases in only 2 out of 12 layers at k=8-10)

## Result Quality

Experiment ran for ~45 minutes (exceeded 30-min cap due to 204800 tokens × 12 layers × 6 k-values
residual collection). Results are clean. Progressive PPL exactly matches cycle 190.

## Key Findings

Frobenius norms at layers 0,2,4,6,8,10 (every other layer):
```
        L0      L2      L4      L6      L8      L10
k=6:  3141.3  4389.2  6043.5  9090.6  13611.3  22038.1
k=7:  3141.3  4389.2  6043.5  9090.6  13942.6  22587.2  (+2.4% at L8, +2.5% at L10)
k=8:  3141.3  4389.2  6043.5  9090.6  13906.7  22892.3  (-0.3% at L8 ← decrease)
k=9:  3141.3  4389.2  6043.5  9090.6  13906.7  23062.7  (+0.8% at L10)
k=10: 3141.3  4389.2  6043.5  9090.6  13906.7  22934.3  (-0.6% at L10 ← decrease)
k=11: 3141.3  4389.2  6043.5  9090.6  13906.7  22934.3  (no change at L10)
```

1. **L0-L6 are identical for k=6..11**: All these blocks are already swapped at k=6. Adding
   more blocks doesn't change the error in already-swapped blocks' outputs.

2. **Only L8 decreases when block 8 is added (k=7→k=8)**: The hybrid activation at block 8
   produces slightly LOWER error at layer 8 than if block 8 had used GELU on the same
   error-contaminated input. This is the "local self-correction" effect.

3. **Only L10 decreases when block 10 is added (k=9→k=10)**: Same mechanism.

4. **All other layers keep growing**: 10/12 layers show no decrease, only the specific
   layer where the newly swapped block sits shows a small decrease.

## Interpretation

The self-correction mechanism is LOCAL: when block l is swapped at step k=l, the hybrid
activation at block l produces slightly lower error at layer l specifically. But this doesn't
help the other layers. The PPL improvement at k=8-10 relative to k=7 must therefore come
from the error ROTATING into less PPL-sensitive directions in the 768-dimensional residual
stream, not from total error reduction.

This is consistent with:
- Cycle 184: diffuse high-rank errors (rank_90=300-450/768). The residual stream has many
  dimensions, and some are more PPL-sensitive than others.
- Cycle 189: cos(e_l, e_{l-1})=0.716 — errors are directionally aligned but not growing
  in ALL directions equally.

## Implication

The self-correction is passive (just error rotation into less harmful dimensions), not
active (not genuine cancellation). The cascade barrier cannot be broken by relying on
self-correction — it's a statistical artifact of the high-dimensional residual stream.

## Scope Note

Experiment exceeded the 30-min cap (~45 min actual runtime). For future experiments using
204800-token residual collection, either reduce N_SEQS or use SEQ_LEN=128.
