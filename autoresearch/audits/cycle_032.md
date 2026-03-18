# Cycle 032 Audit — Item 47: GPT-2 Dead Channel Census v2 (GELU-Correct)

## Audit Status: PASS

## Evidence Review

### Structural dead channels (pre-GELU)
| Threshold | Avg dead (%) |
|-----------|-------------|
| T = -2 | 0.02% |
| T = -3 | 0.00% |
| T = -4 | 0.00% |
| T = -5 | 0.00% |

0 channels are always in GELU's saturated tail at any practical threshold. This confirms cycle 028's finding with the correct metric.

### Near-linear channels
| Threshold | Avg near-linear (%) |
|-----------|---------------------|
| T = +2 | 0.00% |
| T = +3 | 0.00% |
| T = +4 | 0.00% |

0 channels are always in GELU's near-identity regime. GELU channels actively switch between regimes throughout the data distribution.

### Contribution-based analysis
| Threshold | Avg "dead" (%) |
|-----------|----------------|
| < 0.01% of total | 0.18% |
| < 0.05% of total | 93.07% |
| < 0.10% of total | 99.16% |

The high count at 5e-4 and 1e-3 thresholds is misleading: with 3072 channels, equal-share contribution is 1/3072 ≈ 0.033%. Setting threshold at 0.05% is above equal-share, so any moderately uniform distribution will show ~100% "dead" by this metric. The meaningful threshold is 1e-4 (0.01%), where only 0.18% of channels are flagged.

### Smooth pruning curve
| Pruned | PPL | Rel change |
|--------|-----|------------|
| 1% | 128.90 | +1.14% |
| 2% | 132.02 | +3.59% |
| 5% | 143.48 | +12.58% |
| 10% | 194.20 | +52.38% |

Baseline: 127.44 PPL. Even removing the 30 lowest-contribution channels (1% of 3072) increases PPL by >1%. The pruning curve is steep — channels are load-bearing even when individual contribution fractions are small.

## Methodological Note
One script bug was found and fixed: GPT-2 uses `Conv1D` with transposed weight convention, so `c_proj.weight.shape = (3072, 768)`. The original code used `norm(dim=0)` which gives (768,) instead of (3072,). Fixed to use `norm(dim=1)` for this case.

## Verdict Confirmation
**REFUTED** — both criteria met:
1. Structural removable (dead + near-linear) = 0.00% < 1% threshold
2. Pruning 1% → +1.14% PPL, 2% → +3.59% PPL; pruning >2% always causes >1% increase

**Consistent with cycle 028**: GELU FFN channels are all load-bearing. The redesigned GELU-correct thresholds confirm the original cycle 028 finding rather than overturning it. The v2 experiment properly closes this question.

**Key insight**: GPT-2 GELU FFNs have no structural compression headroom. All 3072 channels are active across the data distribution. Compression must operate on the *combination* of channels (rank reduction, channel similarity merging) rather than channel removal.
