# Cycle 225 Review — Item 235 (P0): K=64/32/16 Adam compression cliff sweep

## Verdict: APPROVE

Direct, mechanical extension of cycle 223 (K=128/256) into the small-K regime
{64, 32, 16}. Reuses `experiments/adam_lower_k_cycle223.py` verbatim except
for `K_VALUES`, output paths, and bookkeeping fields. No methodological
delta from cycle 223, which already passed review and ran cleanly.

## What this measures

For each K ∈ {64, 32, 16}: top-k init + lstsq fit of W_out, then 4000 Adam
steps training W_out + b_out only (base frozen), eval every 100, best-state
restore. 3 seeds × 3 K's × BLOCKS=[1,4,7]. Aggregates init/best/best_step/
final mean±std per K. Pre-registered verdict thresholds at 53.0 / 60.0 PPL
on `min_K best_mean`, with INCONCLUSIVE for the (53, 60) band.

## Pre-flight check

- **Disjointness**: K ∈ {16, 32, 64} fully disjoint from cycle 223 (K ∈ {128, 256})
  and cycle 210 (K=512). No re-runs or overwrites — passes
  feedback_autoresearch_experiments rule.
- **Template provenance**: `experiments/adam_lower_k_cycle223.py` exists and
  ran successfully in cycle 223. Mechanical edits only.
- **Param budget per block**: K=64 → 49,920; K=32 → 25,344; K=16 → 13,056.
  Sample budget per block per fit = 200 × 128 = 25,600 calibration tokens.
  MP3 (underdetermined fit): K=64 has 49,920 trainable params vs 25,600 cal
  samples — but this is the **Adam phase**, not lstsq, and the lstsq init is
  on a per-output-channel basis (768 outputs × K trainable each per block →
  per-channel system is K-dim with 25,600 samples, well over-determined for
  all three K values). The Adam phase uses CE on the same 200 calibration
  sequences but is regularized by the topk+lstsq init and best-state restore
  on a held-out 100-sequence eval split. Same regime as cycle 223 — passed.

## Pitfalls swept

- **MP1 (threshold above trivial)**: Float baseline ≈ 53.895; verdict thresholds
  53.0 / 60.0 are tight around float, not above the null. OK.
- **MP3 (underdetermined fit)**: Adam phase, not lstsq closed-form; eval split
  is held out. See preflight note above. OK.
- **MP5 (insufficient capacity)**: This is precisely the question — the cliff
  search exists *because* we expect capacity to fail somewhere. The 3-way
  verdict (CONFIRMED / INCONCLUSIVE / REFUTED) handles all three outcomes
  cleanly, including the "both fail" failure mode (REFUTED at ≥60 PPL).
- **MP6 (refuted on uninformative metric)**: PPL is informative across the
  full 47-200+ range observed in prior cycles. No saturation risk.
- **MP7/MP8**: N/A (not a geometric/dimensional measurement).
- **MP4 (domain transfer)**: Same model, same calib/eval split, same
  optimizer as cycle 223. No domain shift.

## Risks

- **Wall time**: 3 K × 3 seeds × 3 blocks × 4000 steps. Cycle 223 ran 2 K
  on the same surface — this is 1.5× that. Plan includes a fallback (drop
  K=16 if tight). Acceptable.
- **K=16 may be uninformative if K=64 already cliffs**: addressed by the
  fallback rule "do not short-circuit, run full curve for monotone plot".
  Good.
- **Init monotonicity check** (`init_ppl` strictly grows as K shrinks) is in
  Validation. Sensible sanity check.

## Recommendations

None blocking. Plan is mechanical, well-scoped, and the verdict logic is
unambiguous. Approve as written.
