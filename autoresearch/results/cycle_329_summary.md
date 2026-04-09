# Cycle 329 — LASER b11 Spectrum Preflight

**Checkpoint**: `13L_640d_322_7k_wd70_batch20k_best.npz`
**Target block**: b11 (`mlp.proj.weight` shape [640, 1280])
**Gate threshold**: top-13 energy >= 10%

## Spectrum Results

- `proj.weight` shape: [640, 1280]
- n_singular_values: 640
- top-13 energy fraction: **0.068028 (6.80%)**
- Gate pass: **False** (threshold: 10%)
- Rank thresholds: {'0.1': 20, '0.25': 57, '0.5': 136, '0.75': 257, '0.9': 384, '0.95': 458, '0.99': 569}
- S[:10]: [5.4007, 5.1123, 4.8445, 4.7812, 4.7553, 4.7369, 4.7007, 4.6847, 4.6724, 4.6641]

## Verdict

**REFUTED-flat-spectrum**: top-13 energy = 6.80% < 10% gate threshold.

## Comparison to cycles 327-328

| Block | Hidden | Rank | Energy% | Verdict |
|-------|--------|------|---------|---------|
| b3    | 1920   | 5    | 2.60%   | REFUTED-flat-spectrum |
| b12   | 320    | 7    | 7.07%   | REFUTED-flat-spectrum |
| b11   | 1280   | 13   | 6.80%   | REFUTED-flat-spectrum |

**Pattern confirmed across 3 blocks**: output eff_rank is NOT a proxy for weight SVD compressibility.
All three blocks (b3, b12, b11) have flat weight spectra regardless of hidden dim (320-1920).
