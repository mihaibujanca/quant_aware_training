verdict: APPROVE — 0 blocking, 0 advisory
round: 2
fixes_verified: 2/2

## Fix verification

**I1 (null baseline for rank95) — fixed.** P4b adds a random-matrix control: `randn_like(delta)` scaled to matching Frobenius norm, 10 repeats, reports mean ± std of null_rank95. P6 thresholds are now relative to null_rank95 (0.5× for confirmed, 1.0× for refuted). This is the correct comparison — a Gaussian [768, k] matrix has smoothly distributed singular values, so rank95 ≈ 0.95 × min(768, k), establishing the structureless upper bound.

**I2 (residual subspace sample size) — fixed.** P5 now uses 8192 tokens via explicit `max_tokens_per_layer=8192` override. N/d = 10.7 for d_model=768. Top-16 principal directions of the residual covariance are well-estimated at this ratio. The 70% energy threshold in P6 is now reliable.

## R1–R5 checklist

**R1 (meaningful decomposition):** Pass. Core question (lstsq-correction vs new-structure) is directly actionable. GELU stratification explicitly deferred with justification in the Deferred section.

**R2 (null baseline):** Pass. Addressed by P4b. Random-matrix control is sufficient; converged-teacher control would be more informative but is not required for the binary verdict.

**R3 (3 seeds):** Pass. SVD is deterministic given data; seeds control probe sampling and Adam init. 3 is adequate.

**R4 (residual subspace):** Pass. N=8192 resolves the sampling noise concern. N/d=10.7 is comfortable for extracting up to rank95~16 principal directions.

**R5 (known pitfalls):** Pass. MP3 no longer triggered (residual estimation now has N/d=10.7). MP6 addressed by null baseline calibration. MP7/MP8 not triggered.

## Advisory (round 1)

**A1 (GELU stratification dropped):** Resolved. Explicit Deferred section justifies omission and scopes it to a follow-up cycle.
