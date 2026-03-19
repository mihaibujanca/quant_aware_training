# Audit — Cycle 059 (Item 74)

**Title:** Gradient-polished lstsq: 10-50 Adam steps after analytic refit — GPT-2 FFN
**Verdict claimed:** CONFIRMED (38.9% relative improvement at 50 steps, criterion >= 30%)

## Guardrail checks

| # | Check | Status | Notes |
|---|-------|--------|-------|
| G1 | Implementation correctness | PASS | `requires_grad_(False)` freezes all params; only `c_proj.weight` (+bias) unfrozen per block. Loss is next-token CE. |
| G2 | Reproducibility (3 seeds) | PASS (with caveat) | Seeds [42, 123, 7] all run. Per-seed PPL spread is large (seed 42 = 84.9, seed 123 = 67.6 at 0 steps; std ~8.9 PPL). Relative improvement is consistent across seeds (delta_pct_std stable at ~15.3-15.5 across all step counts). |
| G3 | Results interpretation | PASS | 38.9% relative reduction is genuine. Correctly notes +18.4% residual degradation is still impractical. No overclaiming. |
| G4 | Data leakage (polish vs eval) | PASS | Polish uses `probe_holdout` (from train_docs reversed), eval uses `eval_val` (from val_docs). These are from disjoint document pools (80/20 split). No leakage between polish and eval. |

## Key issue

High per-seed variance (PPL std ~8.9, or ~12% of mean) is acceptable for a 3-seed study but makes the absolute PPL numbers unreliable at the individual-seed level. The relative improvement metric (38.9%) is robust because the variance is additive (std barely changes with polish steps), so the signal is clear despite noisy baselines.

## Minor notes

- `probe_train` and `probe_holdout` both draw from `train_docs` (forward vs reversed). With 200 sequences each (25.6K tokens), token overlap is possible if the OWT sample is small. This does NOT affect the audit verdict since the critical separation is polish-vs-eval, which is clean.
- Polishing is sequential across blocks (7, 5, 4, 1). Later blocks see the effect of earlier blocks' polish, creating ordering dependence. Not a bug, but the per-block loss traces are not independent.

## Verdict

**CONFIRMED — audit agrees.** Implementation is sound, data splits are clean, criterion is met with margin.
