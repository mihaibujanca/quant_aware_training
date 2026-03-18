# Audit — Cycle 021

## Item
39 — Grokking as compression catalyst — toy ReLU

## Phase 1: Verdict Check

Claimed: **refuted** — pattern count increases through grokking (369→564→723, no 40% drop), compression gain only +5.7% (need ≥10%).

Criteria: "Refuted if: pattern count is stable throughout training or compression quality doesn't change with training duration."

Evidence: pattern count INCREASES (opposite of hypothesis), compression gain insufficient. Both refuted conditions met.

**Verdict correct.**

## Phase 2: Quality Checks

### G1: Methodology
- Classic grokking setup: n_train=200 (small), weight decay=0.001, long training (30000 epochs). Test accuracy delayed jump confirms grokking occurred.
- Pattern count and entropy measured at regular checkpoints — standard.
- The mechanistic explanation (grokking on continuous tasks = finding more complex solution, not simpler) is insightful and distinguishes this from modular arithmetic grokking studies.

### G2: Reproducibility
Single seed. Training for 30000 epochs is slow (~5min?) but within budget.

### G3: Tautology Check
Not tautological. The increase in pattern count during grokking is a genuine finding.

### G5: Counterintuitive
The INCREASE in patterns during grokking is genuinely counterintuitive relative to the "Deep Grokking" literature (which predicts simplification). But this makes sense for continuous tasks: better test accuracy on spirals requires finer-grained classification boundaries, hence more regions.

## Status: PASS

original_outcome: refuted
audited_outcome: refuted
guardrail_flags: none
reason: Pattern count increases during grokking on continuous 2D tasks. Grokking ≠ geometric simplification in this domain. Clear refutation.
recommendations: The grokking-as-simplification hypothesis may hold for discrete/algebraic tasks but not for continuous classification. This is a domain-dependent effect. Do not assume grokking implies compressibility.
