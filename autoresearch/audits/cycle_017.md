# Audit — Cycle 017

## Item
34 — Region volume Pareto: do large regions carry most task accuracy? — toy ReLU

## Phase 1: Verdict Check

Claimed: **inconclusive** — Gini=0.408 (>0.3, so not refuted), but top 20% covers only 48.2% of correct points (<70%, so not confirmed).

Criteria:
- Confirmed: top 20% covers ≥70%. NOT MET (48.2%).
- Refuted: Gini < 0.3. NOT MET (Gini = 0.408).
- Inconclusive: Pareto holds in 2D but disappears in 100D embedding. Not exactly this case, but the distribution is moderately skewed without reaching the 70% threshold.

**Verdict of inconclusive is correct.** The distribution has mild Pareto structure but not enough for practical compression via region pruning.

## Phase 2: Quality Checks

### G1: Methodology
- 375 distinct patterns on 1000 test points — consistent with cycle 012.
- Gini coefficient computed from region volume fractions — standard.
- Cumulative coverage curve with Hamming fallback — reasonable approach.
- Same network setup as cycles 010, 012 — good consistency.

### G2: Reproducibility
Single seed (42), same as prior cycles. Consistent.

### G3: Tautology Check
Not tautological. The volume distribution depends on training dynamics.

### G4: Scope
Within scope. Tests whether few regions dominate — answer is "somewhat but not enough."

## Status: PASS

original_outcome: inconclusive
audited_outcome: inconclusive
guardrail_flags: none
reason: Volume distribution is moderately skewed but the top quintile doesn't capture enough accuracy for practical compression. Item 35 (GPT-2 analog) remains eligible.
recommendations: The moderate Pareto structure suggests that a more sophisticated routing mechanism (not just keep/discard by volume) might work. But the core question is answered: raw volume-based pruning doesn't give 3x compression. Move on to Procrustes (item 36) for a fundamentally different signal.
