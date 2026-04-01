# Cycle 213 Brief: Adam W_out+W_fc Full Correction (Item 223)

## Result: INCONCLUSIVE

W_out+W_fc (2.36M params) vs W_out-only (1.18M params) for BLOCKS=[1,4,7] K=512:

| Config             | Best PPL | Best Step |
|--------------------|----------|-----------|
| W_out+W_fc         | 46.424   | 1500      |
| W_out-only (c210)  | 46.556   | 4000      |
| W_out-only @1000   | 48.173   | 1000      |

Improvement vs W_out-only@1000: +1.75 PPL (INCONCLUSIVE range 0.5-2.0).
Improvement vs W_out-only best: +0.13 PPL (negligible).

W_out+W_fc reaches the same final quality as W_out-only but in fewer steps (1500 vs
4000). Training W_fc does not provide a fundamental quality advantage. The added compute
(2.36M vs 1.18M trainable params, ~2× slower per step) is not justified.

**Conclusion**: For W_out-only Adam correction at 3-4k steps, W_out-only is preferred
(faster, simpler, same final PPL).
