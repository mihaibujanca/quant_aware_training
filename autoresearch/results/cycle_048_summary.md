# Cycle 048: Local Complexity Proxy as Compression Difficulty Predictor — GPT-2

## Outcome: CONFIRMED

## Setup
- Model: GPT-2 (12 FFN blocks, 768 hidden dim)
- Probe tokens: 128 (fallback from 512 due to compute cost), k=5 neighbors
- Dataset: data/shakespeare.txt (starting at token position len/4)
- 3 Jacobian proxies computed per block: Frobenius std (P1), spectral norm (P2), effective rank (P3)
- Damage ground truth: cycle 005 single-block swap at 8x (hf=0.125) and 16x (hf=0.0625)
- One-tailed Spearman (alternative='greater') throughout

## Per-layer proxy values

| Layer | P1 Frob std | P2 Spec norm | P3 Eff rank | Damage 8x | Damage 16x |
|-------|------------|--------------|-------------|-----------|------------|
|  0    | 12.566     | 33.806       | 72.952      | +17.68%   | +33.12%    |
|  1    | (see JSON) | (see JSON)   | (see JSON)  | +0.25%    | +0.74%     |
| ...   | ...        | ...          | ...         | ...       | ...        |
| 11    | 19.539     | 246.078      | 28.978      | +6.73%    | +6.52%     |

Full values in `cycle_048_local_complexity_gpt2.json`.

## Spearman Correlations (one-tailed, alternative='greater')

| Proxy × Damage | rho_full | p_full | rho_without_L0 | p_without_L0 |
|----------------|----------|--------|----------------|--------------|
| P1 frob_std × 8x  | **0.839** | 0.000 | **0.818** | 0.001 |
| P1 frob_std × 16x | **0.608** | 0.018 | **0.527** | 0.048 |
| P2 spec_norm × 8x | 0.399    | 0.100 | **0.818** | 0.001 |
| P2 spec_norm × 16x| 0.196    | 0.271 | **0.555** | 0.038 |
| P3 eff_rank × 8x  | -0.357   | 0.872 | -0.536    | 0.956 |
| P3 eff_rank × 16x | -0.161   | 0.691 | -0.336    | 0.844 |

## Classification

**CONFIRMED** — P1 (Frobenius std) achieves rho_full=0.839 (8x) and 0.608 (16x), both above 0.5, and rho_without_L0=0.818 (8x) and 0.527 (16x), both above 0.2. The leverage-point guard (rho_without_L0 < 0.2 would trigger INCONCLUSIVE) is not triggered.

## Key Findings

1. **P1 (Frobenius std) is the strongest predictor**: rho=0.839 at 8x, rho=0.608 at 16x. This directly measures variation in the block's local linear behavior across neighboring inputs.

2. **P2 (spectral norm) is a leverage-point dependent**: rho_full=0.399 at 8x (weak), but rho_without_L0=0.818 (strong). The raw spectral norm is dominated by layer 0's outsized damage value. After removing layer 0, spectral norm becomes equally predictive.

3. **P3 (effective rank) anti-correlates**: both rho values are negative. Higher effective rank (more isotropic Jacobians) is associated with *lower* compression damage. This is geometrically intuitive: blocks with isotropic Jacobians are harder to compress because they use many singular directions, but in practice the late/deep layers with high spectral norms have lower effective rank and are harder to compress.

4. **Layer 0 is a genuine hard block, not a leverage artifact**: removing it preserves strong correlation for P1 (0.818 → confirmed). The primary proxy (P1) is robust to the layer-0 outlier.

5. **Practical implication**: Frobenius std of per-token Jacobians is a cheap oracle (no training required, computable in ~5 min on CPU) that predicts where compression will hurt most. The signal generalizes from toy ReLU (cycle 029, rho=0.80 via activation counting) to GELU GPT-2 Jacobian variation.

## Caveats
- Only 128 probe tokens used (fallback). The 512-probe version was computationally infeasible (~48 min vs ~6 min).
- k=5 neighbors (fallback from k=10). Both fallbacks were sanctioned in the plan.
- Damage values are from cycle 005's analytic geometric compression, not a general quantization metric. Generalization to other compression types is untested.
- P2 (spectral norm) result without L0 is strong but requires the leverage-point caveat in interpretation.
