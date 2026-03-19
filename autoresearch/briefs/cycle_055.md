# Cycle 055 Brief: Jacobian Frobenius Std Corpus-Sensitivity Test

## Objective
Test whether the P1 oracle (Jacobian Frobenius std) predicts compression difficulty invariantly across corpora. If corpus-dependent, the oracle must be re-run per target distribution.

## Prior Result (Cycle 048)
- P1 (Frobenius std of per-token Jacobians in k-NN neighborhood) strongly predicts compression damage at 8× (ρ=0.839, p<0.001) and 16× (ρ=0.608, p=0.018)
- All correlations computed on Shakespeare probes only
- Oracle generalization across text domains untested

## Core Question
Does block difficulty ranking (by P1) remain stable when computed on different corpora? If yes: oracle generalizes, one-time cost. If no: corpus-specific computation required.

## Method
- Compute P1 on Shakespeare probes (128 tokens, k=5)
- Compute P1 on WikiText-2 probes (same tokens, same k)
- Spearman rank correlation: P1_shakespeare vs P1_wikitext across 12 blocks
- Confirmed if: Spearman ≥ 0.85 (difficulty ranking corpus-invariant)
- Refuted if: Spearman < 0.5 (ranking fundamentally corpus-dependent)
- Inconclusive: Spearman 0.5–0.85 (partial stability)

## Context
Cycle 044 found severe domain robustness issues: Shakespeare compression (+6%) vs WikiText (+32% corrected PPL). This experiment isolates whether the *oracle* is corpus-specific or just the lstsq refit.

## Key Dependency
Item 71 depends on none. Can proceed independent of other experiments.
