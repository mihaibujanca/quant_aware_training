# Cycle 022 Summary — Attention Head Geometry Census (GPT-2)

## Verdict: INCONCLUSIVE

## Setup
- GPT-2 (12 layers × 12 heads = 144 heads), Shakespeare probe corpus
- 32 probe sequences × 128 tokens
- Metrics: effective rank (nuclear_norm/spectral_norm of mean attn matrix), head output norm variation, pairwise linear CKA within layer
- Prunability score = (1 - eff_rank_norm) × (1 - variation_norm) × mean_CKA_redundancy
- Interventions: zero c_proj rows for top-K heads at K=5, 10, 20, 30

## Key Results

### Geometric census
- 89.6% of heads (129/144) have effective rank < 19.2 (= 0.3 × d_head = 0.3 × 64)
- Effective rank strongly decreases with depth: layers 5–11 show eff_rank concentrating in [1, 7]
- Early layers (0–4) have more variation: eff_rank up to 104 in layer 0 head 6
- Head output variation increases with depth (std range 0.11–0.30 in layer 0, 0.64–4.72 in layer 11)

### Intervention results
| K | PPL | Δ (%) |
|---|-----|--------|
| 0 (baseline) | 125.72 | — |
| 5 | 129.06 | +2.7% |
| 10 | 144.08 | +14.6% |
| 20 | 192.02 | +52.7% |
| 30 | 203.09 | +61.5% |

### Criteria evaluation
- Confirmed criterion: ≥20% low-rank ✓ (89.6%), K=10 ≤5% PPL ✗ (+14.6%)
- Refuted criterion: all high-rank ✗ (most are low-rank), K=5 >15% ✗ (+2.7%)
- Inconclusive criterion: eff rank low but zeroing causes high ppl ✓ → **INCONCLUSIVE**

## Interpretation
GPT-2 attention heads are overwhelmingly low-rank geometrically (89.6% below 0.3×d_head threshold), consistent with "attention is naturally low-rank" findings in the literature. However, this geometric compressibility does not translate into safe pruning:

- Only 5 heads can be zeroed with ≤5% PPL cost
- The 10-head threshold causes +14.6%, implying collective dependencies
- Heads are individually redundant (high CKA with peers) but form a distributed code — no single head's zeroing is catastrophic, but aggregate zeroing is

This mirrors the FFN finding from cycle 018 (Procrustes): geometric linearity/redundancy signals are real but compounding removal is far more destructive than additive individual costs suggest. Attention may benefit more from low-rank weight decomposition (e.g., SVD on W_Q, W_K) than structural head pruning.
