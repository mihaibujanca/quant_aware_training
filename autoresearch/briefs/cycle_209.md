# Cycle 209 Brief: Rank-1 Input FFN Compression (Item 212)

## Result: REFUTED

Block 4 MLP input (after LN2) has rank_90=395, NOT rank-1. The rank-1 assumption from
cycle 198 measured the residual stream before LayerNorm. LayerNorm per-token normalization
spreads the residual stream into ~395 effective dimensions for the MLP input.

Results at block 4:
- Standard K=256: 56.018 PPL
- Standard K=512: 55.715 PPL
- Rank-1 K=256: 58.233 PPL (+2.22 vs standard)
- Rank-1 K=512: 58.207 PPL (+2.49 vs standard)
- Rank-1 K=1024: 58.207 PPL (hits ceiling — more K helps nothing)

The rank-1 model saturates at K≥256: adding more neurons doesn't help because the
single PCA scalar already captures everything the 1D parameterization can express.
Gap exceeds threshold (+2.49 vs limit 2.0). REFUTED.

**Key correction**: The residual stream being rank-1 does NOT mean the MLP input is
rank-1. LN2 normalizes per-token, creating up to 395D effective variation in the
MLP input even when the residual stream is near rank-1.
