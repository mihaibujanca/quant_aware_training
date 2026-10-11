# Cycle 222 Review — Item 227

**Verdict:** APPROVE

## Item
227 (P1) — Frobenius-weighted channel selection: minimize output Frobenius error
on 3-block cascade init PPL (BLOCKS=[1,4,7], K=512), GPT-2 small.

## Plan summary
- Variant A baseline: standard topk by `act_std × ||W_proj[c]||`, 3 seeds
  (42, 137, 2024); reproduces cycle 206 init PPL = 61.746.
- Variant B: deterministic Frobenius-importance score
  `||W_proj[c]||^2 * Σ_t gelu(z_{t,c})^2`, top K=512.
- Variant C (optional): greedy forward selection, dropped if per-block runtime
  >5 min.
- Replace blocks [1,4,7] with each variant's CompressedMLP, fit W_out via lstsq
  on cycle 206 calibration, eval init PPL on cycle 206 eval split.

## Pitfall check (MP1–MP8)
- MP1 (trivial threshold): N/A — comparison is to a measured baseline 61.746,
  not a null distribution.
- MP2 (activation regime mismatch): N/A — both variants score real channels in
  their actual regime; Variant B uses gelu(z) directly, which matches the
  forward path.
- MP3 (underdetermined fitting): N/A — lstsq fits 768×512 W_out from
  200×128 = 25,600 calibration samples per block; n_samples ≫ 2× n_params.
- MP4 (domain mismatch): N/A — same architecture and metric throughout.
- MP5 (capacity): N/A — both variants have identical K=512, same structural
  capacity.
- MP6 (verdict misclassification): the metric (init PPL) is known to vary in a
  meaningful range (cycles 206/213/214 all use it as the optimization target),
  and the verdict thresholds (≤60.75 / ≥0.3 / 0.3–1) are well-defined relative
  to the established 61.746 baseline. No saturation risk.
- MP7 (post-LN measurements): N/A — no geometric measurement; only PPL.
- MP8 (representation vs functional rank): N/A — selection problem, not a
  rank-estimation problem.

## Notes
- Variant B differs from cycle 207's `act_std × W_proj_norm` by using
  Σ gelu(z)^2 instead of std(z), and by squaring the W_proj norm. These are
  related but genuinely different scoring functions, especially because gelu's
  asymmetry matters for sign-dependent weighting.
- The non-identity assertion (Variant B's selection ≠ Variant A on at least one
  block) is the right guard against silently rerunning the baseline.
- Cycle 207 refuted the single-block version. The cycle 222 plan correctly notes
  this is a different metric (cascade init PPL vs single-block PPL) — the
  motivating intuition (cascade amplifies selection differences) is plausible
  enough to justify ~30 min of compute, even if the prior from 207 is low.
- The HALT fallback (if topk fails to reproduce 61.746 within 0.5) is the right
  guard against eval-set drift relative to cycle 206.
- Bookkeeping (move to PAST/Cycle 222, TSV row, STATUS update) is specified.

## Decision
APPROVE — proceed to execute.
