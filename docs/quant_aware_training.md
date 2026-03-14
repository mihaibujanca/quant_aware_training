# Quant-Aware Training: Tomorrow-First Brief

This README is intentionally opinionated and operational.
Use it as the first checkpoint before running new experiments.

## Current Working Direction (Locked For Now)

1. Primary objective: use geometric analysis of quantization error to design **efficient correction** (rank/structure aware), not just to show correction works.
2. Quantization regime for current phase: **weight-only**.
3. Success metric hierarchy:
   - Final: task recovery.
   - Required validation: correction must be shown as **pre-activation recovery**, not extra task capacity.
4. `c_local` is an **analysis upper baseline**, not a deployable target.
5. Low-dimensional experiments are for **intuition + scaffolding** for high-dimensional work.

## Tomorrow: First 30 Minutes Checklist

1. Re-read this file and confirm the working direction still holds.
2. Resolve documentation contradictions listed in "High-Value Gaps".
3. Pick one canonical experiment matrix for current phase (weight-only, classification-first, then transformer).
4. Write down acceptance criteria for "model learned the task enough to analyze."
5. Only then start implementation/experiments.

## High-Value Gaps (Priority Order)

1. No single source of truth for current assumptions.
   - Consequence: experiments and summaries drift in meaning.
2. Contradictory narrative on linear correction.
   - Some docs claim linear is best practical recipe.
   - Current direction is: linear/c_local are baselines to characterize what is fixable.
3. Quantization regime drift (weight-only vs activation/full fake quant).
   - Current phase is weight-only, but docs/notebooks mix regimes in summaries.
4. Weak standardization of benchmark setup.
   - Hard to compare findings across low-dim, classification, transformer.
5. Non-capacity validation is not uniformly enforced.
   - Float ablation / "is this just extra capacity?" checks are inconsistent across experiments.
6. Task-learned gate is unevenly applied.
   - Some analyses run on weakly trained models, which invalidates geometric conclusions.
7. Theory docs are not yet ready to drive experiments.
   - Tropical doc is useful reading, but review explicitly flags overclaims and missing bridges.

## Where the Current Contradictions Live

### A) Linear correction presented as deployment default (needs reframing)

- `/Users/mihai/Projects/quant_aware_training/docs/transformer_correction_sweep.md`
- `/Users/mihai/Projects/quant_aware_training/docs/sweep_results.md`
- `/Users/mihai/Projects/quant_aware_training/docs/quantization_correction_system.md`

### B) Evidence that constrained linear recoverability is weak in current regime

- `/Users/mihai/Projects/quant_aware_training/docs/geometry_overnight_12h_report.md`
  - Q1 not supported under 4-bit constrained correction.

### C) Weight-only focus vs mixed-regime docs

- Weight-only/core geometry direction:
  - `/Users/mihai/Projects/quant_aware_training/docs/geometric_error_correction_brief.md`
  - `/Users/mihai/Projects/quant_aware_training/docs/canonical_quantization_error.md`
- Mixed/full fake quant emphasis:
  - `/Users/mihai/Projects/quant_aware_training/docs/quantization_correction_system.md`
  - `/Users/mihai/Projects/quant_aware_training/notebooks/activation_quantization_geometry.py`

### D) Theory maturity mismatch

- Reading doc with strong claims:
  - `/Users/mihai/Projects/quant_aware_training/docs/tropical_geometry.md`
- Review that flags correctness/scope issues:
  - `/Users/mihai/Projects/quant_aware_training/docs/tropical_geometry_review.md`

## Minimal Acceptance Rules Before Geometry Analysis

1. The model must have clearly learned the task (not random/undertrained behavior).
2. Report task performance metrics in every run artifact.
3. For correction experiments, include a non-capacity sanity check:
   - Apply learned correction to float path and verify it does not act as generic task booster.
4. State quantization regime explicitly in every summary:
   - `weight-only` vs `activation-only` vs `full weight+activation`.

## Immediate Documentation Targets (Do First)

1. Add an "ACTIVE ASSUMPTIONS" section to:
   - `/Users/mihai/Projects/quant_aware_training/docs/geometric_error_correction_brief.md`
2. Add a "PHASE STATUS: READING ONLY, NOT EXPERIMENTAL GROUND TRUTH" banner to:
   - `/Users/mihai/Projects/quant_aware_training/docs/tropical_geometry.md`
3. Add a "REGIME DECLARATION" line to major result docs:
   - `/Users/mihai/Projects/quant_aware_training/docs/sweep_results.md`
   - `/Users/mihai/Projects/quant_aware_training/docs/transformer_correction_sweep.md`
   - `/Users/mihai/Projects/quant_aware_training/docs/quantization_correction_system.md`

## Not Decided Yet (Ask Tomorrow)

1. Policy optimization target: ranking quality vs actual post-training gain.
2. Priority open question among transformer-side hypotheses.
3. Final benchmark matrix once transitioning from classification to transformer/GPT-2 track.
