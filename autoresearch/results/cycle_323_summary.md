# Cycle 323: Post-mlp_scale eff_rank census — REFUTED-scale-matters

**Verdict**: REFUTED-scale-matters

At least one block shifts by ≥50% relative. mlp_scale is doing meaningful rank reshaping; cycle 318 reading should be re-interpreted in the post-scale representation.

**Max relative change**: 85.1%

## Per-block comparison

| Block | Pre-scale (c318) | Post-scale (c323) | Rel change |
|------:|----------------:|------------------:|----------:|
| b00 | 187.61 | 36.35 | -80.6% ← >20% |
| b01 | 160.08 | 123.37 | -22.9% ← >20% |
| b02 | 86.05 | 47.66 | -44.6% ← >20% |
| b03 | 4.89 | 4.01 | -18.1% |
| b04 | 38.30 | 26.74 | -30.2% ← >20% |
| b05 | 39.96 | 37.45 | -6.3% |
| b06 | 28.21 | 27.19 | -3.6% |
| b07 | 23.23 | 20.84 | -10.3% |
| b08 | 30.78 | 30.76 | -0.1% |
| b09 | 28.85 | 26.78 | -7.2% |
| b10 | 21.55 | 16.87 | -21.7% ← >20% |
| b11 | 13.40 | 2.25 | -83.2% ← >20% |
| b12 | 6.94 | 1.03 | -85.1% ← >20% |

## Notes
- Pre-scale: `MLP.__call__` return value (cycle 318 measurement)
- Post-scale: `MLP.__call__` output × `block.mlp_scale` per-channel (cycle 323)
- Checkpoint: `/Users/mihai/Projects/parameter-golf/logs/13L_640d_322_7k_wd70_batch20k_best.npz`
- n_tokens: 25600  seq_len: 1024
