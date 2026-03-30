# Cycle 116 Audit — Extended-Polish Ablation for Cascade Corrections (Item 132)

## Verdict: CONFIRMED — ACCEPTED (with flags)

## G1: Code correctness

- **Conditions A/B/C/D implemented correctly.** A: 500 Adam from fresh compressed model. B: 2500 Adam from fresh compressed model. C: 500 Adam checkpoint + 2000 steps training r=8 corrections only (cycle 115 recipe). D: 500 Adam checkpoint + 2000 more Adam steps on W_down only.
- **C uses cycle 115 recipe.** `LowRankCorrection` with `(I + UV)x`, U=0 init, hooks on block output, CE loss training. Verified identical structure.
- **B and D use same W_down parameters — partially.** D injects 500-step checkpoint weights from A (`_inject_wdown_weights` at line 606) and continues with `_continue_polish_wdown`. However, D creates a **fresh Adam optimizer** at line 276, resetting momentum. B runs one continuous 2500-step optimizer. In practice this doesn't matter: B and D produce nearly identical PPL curves (62.12 vs 62.16).
- **No data leakage.** Polish trains on `probe_holdout` (200 seqs), eval on `eval_val` (200 seqs). Disjoint.

## G2: Seeds

3 seeds [42, 123, 7]. Pattern consistent across all seeds:

| Seed | A (500) | B (2500) | C (500+corr) | D (500+2000) |
|------|---------|----------|---------------|--------------|
| 42   | 70.11   | 73.36    | 69.35         | 73.50        |
| 123  | 53.19   | 54.80    | 52.15         | 54.71        |
| 7    | 56.29   | 58.20    | 55.14         | 58.28        |

All 3 seeds: C < A < B ~ D. Consistent.

## G3: Numeric check

- C=58.88 vs B=62.12: 5.50% gap (>= 5% threshold)
- C=58.88 vs D=62.16: 5.57% gap (>= 5% threshold)
- CONFIRMED per criteria.

## G4: Extended polish degradation — overfitting, not a bug

Learning curves for B and D both show a U-shape:

| Step | B (PPL) | D (PPL) |
|------|---------|---------|
| 500  | 59.86   | 59.86   |
| 1000 | 58.60   | 58.54   |
| 1500 | 59.07   | 58.99   |
| 2000 | 60.23   | 60.13   |
| 2500 | 62.12   | 62.16   |

**Best PPL at step ~1000** (58.5-58.6), then monotonic degradation. This is classic overfitting on the tiny 200-sequence calibration set (25,600 tokens total). By step 2500, each sequence has been seen ~50 times.

The low-rank corrections in C avoid this because: (a) 49K correction params << W_down params per block, so lower capacity, more regularized; (b) corrections operate on the residual stream, not the compressed FFN weights directly.

**Critical observation**: B at step 1000 achieves PPL 58.6, which is 0.3 PPL **better** than C (58.88). If early stopping were applied to B, the correction would show no advantage. The CONFIRMED verdict holds for the specific conditions tested (B at 2500 steps, D at 2500 steps), but the learning curve data reveals the mechanism: corrections win by implicit regularization, not by accessing structure that W_down polish cannot reach.

## G5: What we learned

1. W_down polish overfits past ~1000 steps on 200 calibration sequences, degrading PPL 3.5 points below the 500-step baseline.
2. Low-rank corrections are inherently regularized and avoid this overfitting, giving them a structural advantage at the tested step counts.
3. The advantage is **not** that corrections access fundamentally different recovery — it's that they are harder to overfit. Early stopping of W_down polish (step 1000) would match or beat corrections.
4. This suggests the practical recipe should use either: (a) corrections with 2000 steps, or (b) W_down polish with early stopping around 1000 steps. Both achieve PPL ~58.5-58.9.

## Flags

1. **FLAG: Overfitting confound** — The 5%+ gap between C and B/D is driven by B and D overshooting their optimum. B at step 1000 (PPL 58.6) is competitive with C (58.88). The experiment compares C at its trained endpoint vs B/D at their worst point on the learning curve. This confirms corrections are more robust to step count choice, but not that they access a deeper recovery.
2. **FLAG: Calibration set too small** — 200 sequences (25K tokens) is insufficient for 2500 steps of W_down polish without overfitting. A fairer comparison would use more calibration data or early stopping.
