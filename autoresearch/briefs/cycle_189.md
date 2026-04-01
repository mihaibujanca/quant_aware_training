# Cycle 189: Cascade Error Direction Analysis — GELU vs ReLU+bias Residual Stream

## Hypothesis

The 2.7× superadditivity observed in cycle 181 (ReLU+bias progressive swap) results from **directional error alignment across layers**. When GELU is replaced with ReLU+bias in early blocks, the error propagates through subsequent layers in a coherent, aligned direction rather than diffusing randomly. This directional alignment allows errors to amplify constructively through the network.

**Testable claim**: If errors compound along a consistent direction, successive error vectors e_l and e_{l-1} should exhibit high cosine similarity (> 0.3), enabling potential correction via low-rank interventions along the principal error direction.

## Method

### Setup
- **Model**: GPT-2 (12 transformer blocks, 768 hidden dim)
- **Baseline**: Float model (GELU)
- **Variant**: Progressive ReLU+bias swap — blocks 0..k replaced with ReLU+bias, blocks k+1..11 remain float (k = 0..11)
- **Activation replacement**: GELU → ReLU + per-channel bias (mu_neg, computed from calibration data)
- **Data**: openwebtext_sample.txt, 200 seqs for calibration, 100 seqs for analysis (128 token sequences)

### Measurement Plan

For each progressive configuration (k=0..11):

1. **Forward pass**: Run float model and ReLU+bias model on same 100 analysis sequences (12,800 tokens)
2. **Residual collection**: Install forward hooks on `model.transformer.h[l]` output to capture residual streams at each layer (shape: N_tokens × 768)
3. **Error computation**: e_l = residual_float[l] - residual_swapped[l] for each layer l
4. **Per-layer metrics**:
   - **Directional alignment**: cos(e_l, e_{l-1}) = mean over tokens of normalized dot product
   - **Error magnitude**: ||e_l||_2 (Frobenius norm across all tokens)
   - **Effective rank**: SVD rank at 90% energy threshold of error matrix E ∈ ℝ^{N_tokens × 768}

### Output

JSON file `autoresearch/results/cycle_189_cascade_error_direction.json`:
```json
{
  "progressive_configs": [
    {
      "k": 0,  // blocks 0..k swapped
      "ppl": 15.2,
      "layers": [
        {
          "layer": 0,
          "error_magnitude": 0.45,
          "cos_prev_layer": null,  // null for layer 0
          "effective_rank": 12
        },
        ...
      ]
    }
  ],
  "summary": {
    "mean_cos_aligned": 0.25,
    "mean_cos_all": 0.18,
    "rank_trend": "increasing"
  }
}
```

## Verdict Criteria

| Outcome | Condition | Interpretation |
|---------|-----------|-----------------|
| **Confirmed** | mean cos(e_l, e_{l-1}) > 0.3 in ≥8/12 layers | Errors align directionally; compounding is coherent |
| **Refuted** | mean cos(e_l, e_{l-1}) < 0.1 in all layers | Errors diffuse independently; no directional structure |
| **Inconclusive** | 0.1–0.3 range OR weak trend | Partial alignment; requires follow-up (rotation, magnitude scaling) |

Additional signals:
- If effective rank << 768: low-rank correction plausible
- If rank grows monotonically with layer: error dimensionality increases → harder to correct
- If error magnitude plateaus: saturation; if exponential: uncorrectable

## Expected Outcome

**Likely** (60% confidence): mean cos > 0.3, effective rank 5–15, magnitude grows 2–5×. This explains superadditivity: errors compound along principal component(s), amplifying task-relevant distortions.

**Alternative** (30% confidence): cos ≈ 0.1–0.2, rank ≈ 50+, magnitude growth moderate. Errors partially aligned but not strongly; superadditivity driven by magnitude scaling alone, not directional coherence.

**Unlikely** (10% confidence): cos < 0.05, rank ≈ 500+, random growth. No directional pattern; superadditivity is task-specific artifact.

## Runtime & Resources

- **Calibration**: ~3 sec (200 seqs, hook overhead)
- **Analysis**: ~15 sec (100 seqs × 12 progressive configs, hook overhead)
- **Computation**: ~5 sec (SVD, cosine similarity, aggregation)
- **Total**: ~25 sec on MPS/CPU
- **Device**: MPS (macOS GPU) or CPU fallback
- **Output size**: ~50 KB JSON

## Success Criteria

Experiment is **valid** if:
1. Progressive PPL matches cycle 181 table (within ±2%)
2. Error vectors have non-trivial rank (> 2, < 768)
3. All metrics computed without NaN/Inf
4. JSON fully written and parseable

Verdict is **actionable** if mean cos trend is clear (> 0.2 or < 0.1 in most layers).
