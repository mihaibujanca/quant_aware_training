# Cycle 056 Audit — OWT Compression Ratio Sweep

**Item**: 72 — Compression ratio sweep on OpenWebText: finding the viable frontier — GPT-2 FFN
**Claimed verdict**: CONFIRMED
**Audit verdict**: REFUTED — non-monotonic pattern is an lstsq overfitting artifact

## Guardrail Flags

| Check | Status | Note |
|-------|--------|------|
| G1: Implementation correctness | **FAIL** | Channel counts correct, but lstsq probe budget insufficient for 2× |
| G2: Reproducibility (3 seeds) | PASS | 3 seeds [42, 123, 7], consistent across seeds |
| G3: Results interpretation | **FAIL** | "2× worse than 4×" presented as genuine finding; it is a methodology artifact |
| G4: Counterintuitive results | **FAIL** | Non-monotonicity not investigated; root cause is lstsq underdetermination |
| G5: Method failure vs genuine | **FAIL** | This is method failure, not a finding |

## Root Cause

The lstsq output projection refit uses `max_tokens_per_layer=2048` probe tokens for ALL ratios. The design matrix dimensions are:

| Ratio | Kept channels | lstsq columns (k+1) | Overdetermination ratio |
|-------|--------------|---------------------|------------------------|
| 2× | 1536 | 1537 | 1.33× |
| 3× | 1024 | 1025 | 2.00× |
| 4× | 768 | 769 | 2.66× |
| 6× | 512 | 513 | 3.99× |

At 2× compression, the lstsq system is barely overdetermined (1.33×). The fit memorizes the 2048 probe tokens and generalizes poorly to the 200 eval sequences. At 4× and 6×, there is much more regularization from overdetermination, so the fit generalizes better. This fully explains the non-monotonic pattern: 2× has the most free parameters relative to data, hence the worst generalization.

## Recommendations

1. **Increase `max_tokens_per_layer`** to at least 4096 (ideally 8192) so that even 2× has overdetermination ≥ 2.5×.
2. **Add L2 regularization** (ridge regression via `torch.linalg.lstsq` driver or manual Tikhonov) to the output projection refit. This would make results robust across ratios regardless of probe count.
3. **Re-run** with fixed probe budget before claiming non-monotonicity. The "viable frontier" conclusion is premature.
4. **Do NOT mark item 72 as CONFIRMED** — the experiment does not validly test what it claims.
