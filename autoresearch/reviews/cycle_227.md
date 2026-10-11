# Cycle 227 Review — Item 237 (P0)

## Verdict: APPROVE

## Summary

Plan extends the cycle 218 centred Δ̃_l → c_fc(j) alignment measurement
from the adjacent destination only (j=l+1) to ALL j > l. This is the
specificity control flagged by the cycle 218 audit (G3): without it, the
+17 to +33σ adjacent result could reflect a generic universal reading
basis rather than block-to-(l+1) inter-block communication.

## Checks

- **Pipeline reuse**: builds on `experiments/delta_centred_alignment_cycle218.py`
  (verified working, audit PASS in cycle 218). Reuses centred Δ̃_l
  computation, c_fc reader, top_left_svs helper, random_orthonormal
  baseline. Drops unused readers (Q/K/V, residual PC, vocab interp).
- **Hypothesis verbatim**: matches item 237 exactly. Same N_CALIB=200,
  SEQ_LEN=128, SEED=42, 10-seed Gaussian QR baseline.
- **Verdict criteria**: well-specified —
  - CONFIRMED: median_specificity ≥ 1.0 AND n_rows_with_quiet_nonadj ≥ 7
  - REFUTED: median_specificity < 0.3
  - INCONCLUSIVE: 0.3 ≤ median_specificity < 1.0
  Specificity = row_max_at_lp1 − row_mean_nonadj defined unambiguously.
  l=10 row correctly excluded from row-quietness count (no non-adjacent j).
- **Sanity checks**: adjacent (l, l+1) cos_sum_top20 must reproduce cycle
  218's 4.42–7.07 range; random baseline cos_sum_top20 must match cycle
  218's ~2.72±0.10. Hard-stops if >5% deviation. Good guardrail against
  silent data-loader/hook drift.
- **Compute**: ~1 minute, single forward pass over 200×128 calib + SVDs.
  Trivially under 30 min budget.
- **Random baseline**: reused unchanged (10 seeds Gaussian orthonormal
  20×768). Fallback to 5 seeds documented if needed.
- **Marginal-case fallback**: if median_specificity ∈ [0.8, 1.2] OR
  n_rows_with_quiet_nonadj ∈ {6,7,8}, report exact numbers and pick the
  strict-criterion verdict. Sensible.
- **Bookkeeping**: agenda move (NEXT P0 → PAST cycle 227), TSV append,
  STATUS.md, conventional commit message — all listed.

## Notes

- The (l, j) heatmap with l < j has 11+10+...+1 = 66 cells, of which
  11 are adjacent and 55 are non-adjacent — large enough sample for the
  per-row mean to be stable, except row l=10 (which has zero non-adjacent
  entries) and row l=9 (only one non-adjacent entry, j=11 — high variance
  but the median over 10 rows absorbs it).
- Plan stores both top5 and top20 cos sums; top5 is the verdict metric,
  top20 is for cross-check against cycle 218. Good redundancy.
- The specificity definition (max minus mean) is the right contrast: a
  generic-reading-basis null predicts row_max_at_lp1 ≈ row_mean_nonadj,
  whereas a true inter-block-adjacency null predicts a sharp peak at l+1.

No blocking concerns. Approve as-is.
