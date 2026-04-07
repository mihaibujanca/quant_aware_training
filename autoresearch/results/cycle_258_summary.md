# Cycle 258 — Pullback κ/S: GELU vs ReLU+b_leak vs gelu_noneg+μ (item 269)

**Verdict: CONFIRMED**

For three variants of GPT-2 small on the standard 200×128 wikitext-2 calibration slice
(seed 42), collect the residual stream trajectory at all 13 layer boundaries via
`aleph.pullback_geometry.collect_residual_trajectory`, and compute pullback metric
salience S_l and curvature κ_l per token under G = U^T U.

| Variant | calib PPL | ρ(S, κ) over layers 1..11 |
|---|---|---|
| A — GELU baseline | 45.064 | -0.3801 |
| B — ReLU + per-block per-channel b_leak | 60.082 | -0.3801 |
| C — gelu_noneg + per-channel μ (cycle 228) | 61.664 | -0.3876 |

**κ profile (mean per layer 1..11)**

| layer | A_gelu | B_relu+b_leak | C_gelu_noneg+μ | rel ΔReLU | rel Δnoneg |
|---|---|---|---|---|---|
| 1 | 0.0001 | 0.0001 | 0.0001 | 0.118 | 0.002 |
| 2 | 0.0022 | 0.0026 | 0.0021 | 0.144 | 0.074 |
| 3 | 0.0022 | 0.0030 | 0.0024 | 0.357 | 0.060 |
| 4 | 0.0019 | 0.0027 | 0.0019 | 0.477 | 0.013 |
| 5 | 0.0011 | 0.0015 | 0.0012 | 0.408 | 0.065 |
| 6 | 0.0007 | 0.0008 | 0.0007 | 0.261 | 0.070 |
| 7 | 0.0003 | 0.0004 | 0.0003 | 0.133 | 0.050 |
| 8 | 0.0002 | 0.0002 | 0.0002 | 0.037 | 0.104 |
| 9 | 0.0001 | 0.0001 | 0.0001 | 0.028 | 0.100 |
| 10 | 0.0000 | 0.0000 | 0.0000 | 0.118 | 0.035 |
| 11 | 0.0000 | 0.0000 | 0.0000 | 0.314 | 0.505 |

**Verdict logic.**
- early-band (layer 1..6) max |Δκ|/κ_GELU for ReLU = **0.4772** (need ≥0.30 for CONFIRMED clause 1)
- |ρ_ReLU(S, κ)| = **0.3801** (need ≤0.50 for CONFIRMED clause 2)
- all-layer max |Δκ|/κ_GELU for ReLU = **0.4772** (need <0.10 for REFUTED clause 1)
- ρ_ReLU = **-0.3801** (need ∈ [−0.85, −0.65] for REFUTED clause 2)
- Result: **CONFIRMED**

Plot: `autoresearch/results/cycle_258_kappa_salience.png`. Per-token κ/S aggregates (mean / p10 / p50 / p90) saved in JSON.
