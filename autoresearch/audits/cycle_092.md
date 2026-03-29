# Audit — Cycle 092

- **Item**: 107 — Adam polish dissection — GPT-2 FFN
- **Cycle**: 092
- **Auditor verdict**:

```
status=UNCERTAIN
old_outcome=inconclusive
new_outcome=inconclusive
reason=The INCONCLUSIVE verdict is technically correct under the stated criteria, but both metrics (rank_ratio and residual_frac) have methodological issues that make the inconclusive zone unreliable. (1) rank95 computed on singular values, not squared singular values (energy), inflating apparent rank and compressing the ratio toward 1.0. (2) residual_frac uses rank95(delta) to truncate the residual basis, coupling the two metrics and inflating residual_frac toward the random baseline of rank95/d_model ~ 0.508. The observed 0.58-0.62 is only 7-11% above random alignment. (3) Per-seed eval uses different data splits, making PPL comparisons across seeds invalid (seeds 123 and 7 appear to beat the teacher, but evaluated on easier splits). The experiment is well-structured but the measurement apparatus is too blunt to distinguish "diffuse update" from "structured update masked by insensitive metrics."
guardrail_flags=G1,G3,G6
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Re-run with rank95 on sigma^2 (standard energy metric) and fixed-dimension residual projection (use rank95 of residual, not rank95 of delta). Fix eval to use same data split across seeds. If energy-based rank_ratio drops below 0.5, the verdict would flip to CONFIRMED.
interesting=true
interesting_reason=The sigma vs sigma^2 discrepancy is large: on a synthetic rank-10+noise matrix, sigma gives ratio=0.60 while sigma^2 gives 0.027. The current metric cannot distinguish a structured delta from a random one. This may retroactively affect interpretation of any prior cycle that used _compute_rank95.
```

## Guardrail Details

**G1 (Implementation)**: FLAGGED — two measurement issues.

(a) `_compute_rank95` uses `S.cumsum(0) / S.sum()` — cumulative sum of singular values, not squared singular values. Standard "rank at 95% energy" uses sigma^2. The sigma metric is far less sensitive to low-rank structure. Verified empirically: a rank-10+noise matrix has rank95(sigma)=263/512 (ratio=0.60) but rank95(sigma^2)=10/512 (ratio=0.027). The null baseline uses the same metric, so the ratio is self-consistent, but it compresses all ratios toward 1.0 and cannot detect moderate structure.

(b) `_residual_projection` truncates `U_res` to `rank95` columns, where `rank95` is from the delta SVD, not from the residual SVD. This means projecting onto a 390-dim subspace of 768-dim space, which gives ~50.8% overlap for any random vector. The observed 58-62% is only marginally above random. The correct approach would use the residual's own effective rank (or compute a random-direction baseline for the specific dimensionality used).

Both issues inflate the "full-rank, mixed-subspace" appearance. The delta may have more structure than detected.

**G2 (Reproducibility)**: PASS. 3 seeds (42, 123, 7), 4 blocks, all hyperparameters explicit. Per-seed raw values stored in JSON. rank95 is remarkably stable across seeds (std=0 for blocks 7 and 5, std=0.577 for blocks 4 and 1). residual_frac std < 0.009 across seeds. The experiment is well-seeded and deterministic.

**G3 (Results)**: FLAGGED — PPL comparisons across seeds use different eval data splits. Teacher PPL (57.65) is evaluated on shuffle_seed=42 corpus. Per-seed students are evaluated on shuffle_seed={42,123,7} corpora. This makes seed-averaged PPL (72.6→58.1) an average over different eval sets. Seeds 123 (51.5) and 7 (54.6) appear to beat the teacher at step 500, which is almost certainly an artifact of easier eval splits, not genuine improvement. The convergence *trend* (PPL decreasing monotonically) is valid within each seed.

**G4 (Surprising?)**: FLAGGED — rank95=390/512 (76% of min dimension, 89% of null) is presented as "nearly full-rank." With the energy metric, this could be substantially lower. The high rank under the sigma metric should not be interpreted as evidence of a diffuse update without first checking the energy metric. The leading singular value (1.58 at block 7) is 2× the second (0.88), suggesting some concentration that the sigma metric masks.

**G5 (Learning quality)**: ADEQUATE as a first-pass exploration, but the inconclusive result is not actionable without fixing the measurement issues. The experiment asks a good question (is polish low-rank residual correction?) but the metrics cannot answer it reliably.

**G6 (Method)**: FLAGGED — the coupling between rank95(delta) and residual_frac creates a dependency: a higher-rank delta automatically samples more of the residual subspace, inflating residual_frac. This means the two criteria (rank_ratio < 0.5 AND residual_frac > 0.70) are not independent. A delta that is truly low-rank (rank_ratio < 0.5) would use fewer residual components (say 200/768), making it *harder* to reach residual_frac > 0.70 even if it is perfectly aligned with the residual subspace. The thresholds are calibrated against coupled metrics. The 8192-token sample (N/k=16) is adequate for lstsq but marginal — not a primary concern.
