# Cycle 340 Review

## Plan quality: PASS

### Checklist
- [x] Reuses existing checkpoints (no retraining needed)
- [x] Same calib data pipeline as cycle 332 (200 sequences, wikitext-2)
- [x] Hooks c_proj output (same measurement point as cycle 332)
- [x] exp(entropy) formula is correct: p_i = λ_i/Σλ, eff_rank = exp(-Σ p_i ln p_i)
- [x] Compares all 3 seeds per activation (not just seed 42)
- [x] Verdict thresholds match agenda item

### Risks
- Near-zero eigenvalues: need epsilon clamp in p_i log p_i to avoid NaN. Plan doesn't mention this explicitly but implementation should handle it.
- MPS eigendecomposition can be flaky — use CPU for eigh.

### Estimate
- ~30 seconds wall time (no training, just forward passes + eigendecomp)
