# Audit — Cycle 086

- **Item**: 99 — FFN output manifold geometry — GPT-2 FFN
- **Cycle**: 086
- **Auditor verdict**:

```
status=PASS
old_outcome=inconclusive
new_outcome=inconclusive
reason=Implementation correct: eigenspectrum via centered covariance + eigh, GELU applied post-hook on c_fc, JL distortion properly scaled by sqrt(d/k). Isotropy ratios (0.004-0.016) genuinely land in the inconclusive 0.01-0.1 band for 3/4 blocks. Verdict criteria applied faithfully. JL tightness (3-4x) is the expected gap between worst-case union bound and empirical concentration — not a novel finding.
guardrail_flags=none
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Inconclusive means the experiment did not discriminate. Consider: (a) refining the isotropy metric — lambda_512/lambda_1 is crude; effective dimension (participation ratio = (sum lambda)^2 / sum lambda^2) would give a single scalar, (b) testing whether lstsq dominance correlates with something more specific than bulk isotropy (e.g., alignment between top-k eigenspace and the quantization error subspace), (c) closing this line — moderate isotropy is the expected answer for a trained transformer, and the question "why does lstsq work" may be better answered by studying the error subspace directly rather than the ambient manifold geometry.
interesting=false
interesting_reason=All findings are expected: moderate isotropy in trained transformers, JL bounds loose by standard concentration gap, MP departure from structured data. No surprising observations.
```

## Guardrail Details

**G1 (Implementation)**: PASS. Eigenspectrum: centered covariance `(X_c^T X_c) / (n-1)`, symmetrized, `eigh` returns ascending order, flipped to descending. GELU hook on `c_fc` (pre-GELU linear) applies `F.gelu(..., approximate="tanh")` — matches GPT-2's activation. JL distortion: `|d_proj/d_orig - 1|` with `sqrt(3072/512)` scaling for orthonormal projections. JL bound `eps = sqrt(8*log(n)/k)` is the standard formula. All correct.

**G2 (Reproducibility)**: PASS. Seeds: `default_rng(42)` for MP sampling, `default_rng(123)` for JL pairs, `torch.manual_seed(seed_i * 7 + 13)` for random projections. No stochastic training. Probe corpus from 4 deterministic text sources. 10 random projection draws for variance estimation.

**G3 (Results interpretation)**: PASS. Isotropy ratios: Block 7=0.0151, Block 5=0.0157, Block 4=0.0151 (all in 0.01-0.1), Block 1=0.0043 (below 0.01). Confirmed requires >0.1 for 3/4 blocks (0 meet this), refuted requires <0.01 for 3/4 blocks (only 1 meets this). INCONCLUSIVE is the correct verdict. The agenda writeup accurately reports these numbers and correctly notes the top-k SVD over-compensation (max distortion 1.4 due to non-isometric sqrt(d/k) scaling on variance-concentrated directions).

**G4 (JL tightness — artifact or genuine?)**: Not an artifact, but also not interesting. The JL lemma uses a union bound over all n^2 pairs, giving `log(n)` in the exponent. The experiment samples only 10K of ~134M possible pairs, and concentration is naturally tighter than the worst-case bound. This 3-4x gap is a textbook observation. It would be surprising if the gap were <1.5x or >10x.

**G5 (Learning quality)**: LOW. The experiment lands squarely in the inconclusive zone and produces no actionable discriminator. The MP departure (53-72% outside support) confirms structure beyond random — expected for a trained model, not informative. No clear next step emerges from these results beyond "the spectrum is moderately steep," which was already the default assumption.

**G6 (Finite-sample effects)**: Minor concern but does not affect verdict. With gamma = d/n = 3072/16384 = 0.1875, the top eigenvalue has ~10-20% upward bias (Tracy-Widom / BBP transition), which would slightly inflate the denominator of the isotropy ratio, making it appear more anisotropic than reality. Correcting for this would push ratios slightly upward (maybe to 0.017-0.019), still firmly in the inconclusive band. The experiment does not mention or correct for finite-sample bias, but it doesn't need to given the margin.
