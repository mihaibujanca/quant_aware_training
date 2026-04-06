# Cycle 228 — gelu_noneg with per-channel first-moment correction (item 238)

**Verdict: INCONCLUSIVE**

Cycle 220's gelu_noneg patch (clip negative leak in all 12 blocks) gave PPL=2093
(+4545%), audited as a missing-mean-shift artefact. This cycle precomputes the
per-channel mean leak μ_l ∈ R^3072 on the same N=200 wikitext-2 calibration split
and adds it as a post-activation bias. Same eval split as cycle 220 (N=200,
SEQ_LEN=128, seed 42).

| Variant | PPL | Δ vs baseline | % vs baseline |
|---|---|---|---|
| Float baseline | 45.064 | — | — |
| gelu_noneg (cycle 220, uncorrected) | 2093.296 | +2048.232 | +4545% |
| gelu_noneg + per-channel μ (cycle 228) | 61.664 | +16.600 | +36.84% |

Thresholds: low=49.57 (×1.10), refute=100.00.

**μ statistics (per block aggregate).** Overall mean(μ) ≈ -0.08744,
overall mean(|μ|) ≈ 0.08744. μ is ≤ 0 elementwise by construction
(GELU(z) − relu(GELU(z)) clips only negatives).

**Conclusion.** Per-channel first-moment correction recovers most of the cycle 220 gap (2032 PPL) but lands in the INCONCLUSIVE band (49.57 < 61.66 < 100). First-moment correction explains most but not all of the cycle 220 catastrophe.
