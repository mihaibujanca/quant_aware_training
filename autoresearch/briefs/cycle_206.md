# Cycle 206 Brief: Adam W_out Correction — BLOCKS=[1,4,7] K=512 (Item 217)

## Result: CONFIRMED

Adam correction on W_out/b_out only (1.18M params, frozen fc weights) drops PPL
from 61.746 (topk+lstsq init, +14.6% vs float) to 48.111 — which is **10.7% below
float** (53.895). Improvement: 13.635 PPL vs target 2.75.

Eval PPL still decreasing at step 1000. No overfitting.

**Key insight**: The cascade barrier is not structural. It is from suboptimal W_out
initialization. The original W_out was fit via lstsq to approximate the full 3072-dim
MLP output; Adam finds a W_out that works optimally for the 512-channel compressed
space. 1.18M parameters entirely dissolve the +14.6% cascade.
