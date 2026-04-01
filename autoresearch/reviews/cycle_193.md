# Design Review — Cycle 193 (Item 203): C3+Hybrid Combined

## Correctness

**R1 — Combined activation semantics**:
- z<0: mu_neg (per-channel mean of GELU(z) for z<0 on calibration data) — same as C3
- z>=0: ReLU(z) — same as Hybrid
This is the union of both modifications. Neither regime is preserved. **CORRECT**.

**R2 — mu_neg calibration**: Same as cycle 188 C3 experiment. mu_neg[b] is the per-channel
mean of GELU output when z<0, computed from calibration data. Applied as a constant for
all z<0 tokens. **CORRECT — consistent with cycle 188**.

**R3 — Not equivalent to ReLU+bias**:
- ReLU+bias for z<0: max(0, z) + bias = 0 + bias. The bias was calibrated to approximate
  E[GELU(z)] overall, not just E[GELU(z)|z<0].
- C3+Hybrid for z<0: mu_neg (= E[GELU(z)|z<0] per channel).
- These are different constants for z<0. This is the correct distinction. **CORRECT**.

**R4 — Config matches cycle 188**: SEQ_LEN=1024, N_SEQS=200, SEED=42. Baseline ~22.700.

## Potential Issues

**P1 — Verdict thresholds**: The plan says confirmed if >98% increase. ReLU+bias is +88%.
To confirm "general superadditivity exceeding ReLU+bias", the threshold should be >88%.
Using >98% is more conservative (requires C3+Hybrid to be WORSE than ReLU+bias by 10%).

Actually, a more informative threshold: if C3+Hybrid > 45 PPL (+98% over 22.7), confirmed.
If C3+Hybrid < 35 PPL (+54%, below the sum of C3+Hybrid ≈ 22.7*1.48=33.6), refuted.
The boundary at 35 is the expected "no superadditivity" outcome.

Actually: 22.700 × (1 + 0.2033 + 0.2767) ≈ 22.700 × 1.48 = 33.6. So refuted threshold
should be <34, not <28. Let me recalculate: baseline=22.700, 48% more = 33.6.
Plan says refuted if "28-35" which brackets 33.6 — fine.

**P2 — Progressive sweep cost**: 12 deepcopies × SEQ_LEN=1024 × 200 seqs = ~36 forward passes.
At ~3s per pass on MPS: ~108s. Within 30 min. **OK**.

## Flags

None blocking.

**REVIEW RESULT: APPROVED.**
