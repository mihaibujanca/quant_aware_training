# Cycle 196 Brief: LayerNorm Recalibration for Stacked Compressed Blocks (Item 207)

## Item

207 — LayerNorm recalibration — lightweight cascade fix for GPT-2 topk+lstsq

## Context

Hypothesis: cascade barrier arises partly because each compressed block shifts the residual
stream distribution, and subsequent LN layers receive shifted inputs. If LN γ/β parameters
are adjusted to compensate for the shift in per-channel statistics, the normalized outputs
might better match what downstream layers expect.

## Result

Compressed baseline (blocks [1,4,7,11], K=512, 6x): PPL 30.089
LN-2 recal (compressed blocks only): PPL 30.337 (WORSE by 0.248)
LN-2 recal (all 12 blocks): PPL 30.645 (WORSE by 0.556)
Best improvement: -0.248 (no improvement)

REFUTED.

## Interpretation

LN operates per-token (normalizes each token vector over 768 channels independently).
Per-channel cross-token statistics (what I calibrate to) are irrelevant to how LN
normalizes individual tokens. The per-channel gamma/beta adjustments interfere with
the already-working per-token normalization, making things slightly worse.

The cascade barrier is not from LN distributional mismatch — it's from the modified
residual content flowing into subsequent layers, which LN cannot fully compensate for.
