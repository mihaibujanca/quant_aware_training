# Cycle 302 — Item 312: Unified curvature residual geometric framework

**Verdict: CONFIRMED**

(a) n_blocks rank_95 ≤ 64 = 9/12 (pass ≥9: True), n within ±5% = 12/12; (b) n_pairs c_fc_top5 ≥ 2.0 = 11/11 (pass ≥6: True); (c) max κ shift (layers 3–5) = 0.4772 (pass ≥0.35: True).

## Framework
Module `experiments/curvature_residual_geometric.py::UnifiedCurvatureFramework` parameterises the curvature residual Δy = W_proj · (sigma − sigma_lin)(z) for any activation pair. This run: sigma=GELU (tanh approx), sigma_lin=ReLU, GPT-2 small, calib = 200 wikitext-2 val seqs × 128 tokens = 25600 tokens, seed 42.

## Panel 1 — raw Δy rank_95 (cycle 217 reproduction)

| block | this cycle | cycle 217 ref | |diff|/ref |
|---:|---:|---:|---:|
| 0 | 10 | 10 | 0.000 |
| 1 | 132 | 132 | 0.000 |
| 2 | 1 | 1 | 0.000 |
| 3 | 1 | 1 | 0.000 |
| 4 | 1 | 1 | 0.000 |
| 5 | 1 | 1 | 0.000 |
| 6 | 1 | 1 | 0.000 |
| 7 | 8 | 8 | 0.000 |
| 8 | 73 | 73 | 0.000 |
| 9 | 110 | 110 | 0.000 |
| 10 | 1 | 1 | 0.000 |
| 11 | 1 | 1 | 0.000 |

n blocks rank_95 ≤ 64 = **9/12** (ref 9/12). n within ±5% of c217 reference = **12/12**.

## Panel 2 — centred Δ̃ c_fc top-5 cos sum (cycle 218 reproduction)

| l→l+1 | c_fc top5 | Q top5 | K top5 | V top5 |
|---|---:|---:|---:|---:|
| 0→1 | 2.782 | 2.331 | 2.532 | 1.977 |
| 1→2 | 2.705 | 2.456 | 2.150 | 2.203 |
| 2→3 | 2.586 | 1.985 | 1.935 | 2.382 |
| 3→4 | 2.701 | 1.656 | 1.837 | 2.470 |
| 4→5 | 3.318 | 2.137 | 1.752 | 2.391 |
| 5→6 | 3.233 | 2.155 | 2.355 | 2.648 |
| 6→7 | 3.327 | 2.184 | 1.599 | 2.107 |
| 7→8 | 3.356 | 2.272 | 1.812 | 1.824 |
| 8→9 | 3.393 | 2.451 | 1.705 | 0.840 |
| 9→10 | 2.762 | 2.137 | 1.358 | 0.724 |
| 10→11 | 2.493 | 1.763 | 1.361 | 0.820 |

n pairs c_fc_top5 ≥ 2.0 = **11/11** (ref 11/11).

## Panel 4 — E_kept_top2 vs Vh[:2] of W_proj (cycle 301 reproduction)

| block | E_kept_top2 | cycle 301 ref |
|---:|---:|---:|
| 0 | 0.5902 | 0.5902 |
| 1 | 0.7463 | 0.7463 |
| 2 | 0.9190 | 0.9190 |
| 3 | 0.8649 | - |
| 4 | 0.8426 | - |
| 5 | 0.8109 | - |
| 6 | 0.7197 | - |
| 7 | 0.6622 | - |
| 8 | 0.4722 | - |
| 9 | 0.5356 | - |
| 10 | 0.8374 | - |
| 11 | 0.9803 | 0.9803 |

## Panel 5 — pullback κ shift under sigma_lin + b_leak swap (cycle 258 reproduction)

| layer | κ_A (GELU) | κ_B (ReLU+b_leak) | rel shift | c258 ref shift |
|---:|---:|---:|---:|---:|
| 1 | 0.00006 | 0.00007 | 0.118 | 0.000 |
| 2 | 0.00223 | 0.00255 | 0.144 | 0.182 |
| 3 | 0.00222 | 0.00302 | 0.357 | 0.364 |
| 4 | 0.00185 | 0.00274 | 0.477 | 0.421 |
| 5 | 0.00109 | 0.00154 | 0.408 | 0.364 |
| 6 | 0.00067 | 0.00084 | 0.261 | 0.143 |
| 7 | 0.00034 | 0.00038 | 0.133 | 0.333 |
| 8 | 0.00019 | 0.00020 | 0.037 | 0.000 |
| 9 | 0.00010 | 0.00011 | 0.028 | 0.000 |
| 10 | 0.00002 | 0.00003 | 0.118 | 0.000 |
| 11 | 0.00000 | 0.00001 | 0.314 | 0.000 |

max rel κ shift in layers 3–5 = **0.4772** at layer 4 (ref 0.477 at layer 4).

## Verdict logic (item 312 verbatim)
- CONFIRMED iff (a) n_blocks rank_95 ≤ 64 ≥ 9 AND per-block within ±5% of cycle 217 reference, AND (b) n_pairs c_fc_top5_cos_sum ≥ 2.0 ≥ 6, AND (c) max κ shift in layers 3–5 ≥ 0.35.
- REFUTED iff any reproduction off by > 2× (n_rank95_le_64 < 5, n_cfc_ge_2 < 3, or max_kappa_shift < 0.18).
- INCONCLUSIVE otherwise.

**Result: CONFIRMED**.

## Artifacts
- experiments/curvature_residual_geometric.py
- experiments/unified_curvature_cycle302.py
- autoresearch/results/cycle_302_unified.json
- autoresearch/results/cycle_302_panels.png
- autoresearch/results/cycle_302_summary.md
