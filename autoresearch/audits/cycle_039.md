status: PASS
original_outcome: confirmed
audited_outcome: confirmed
guardrail_flags: G2 (single seed)
reason: k=5 correction recovers 3.6pp (≥3pp threshold) and k=4 is not worsened (+1.23pp improvement). The lstsq system is heavily overdetermined (25600 tokens × 769 augmented features → 768 targets). MSE reduction of 20.8% is modest but translates to meaningful PPL recovery. The high correction rank (392/768 for 95% energy) indicates the error is distributed, consistent with it being a cumulative effect of 5 independently-compressed blocks. Single seed is the only flag but the 3.6pp recovery is well above the 3pp threshold.
recommendations: Combine correction with cycle 037's budget analysis: k=5 with correction at +12.53% vs k=4 without correction at +9.59%. The gap is now 3pp instead of 7pp. Try multi-point corrections (one after each compressed block cluster) to see if recovery can be pushed further. Also test whether correction helps at k=6.
