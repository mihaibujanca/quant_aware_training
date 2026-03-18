# Audit — Cycle 010

## Item
20 — Single-affine collapse quality — toy ReLU

## Phase 1: Verdict Check

Claimed outcome: **confirmed** — 3/4 hidden layers have r < 0.15.

Agenda criteria: "Confirmed if: at least one hidden layer achieves r < 0.15 on held-out probe data."

Evidence: L2 r=0.050, L3 r=0.033, L4 r=0.024 — all well below 0.15. Only L1 (r=0.43) is above.

**Verdict is correct.** The criterion requires "at least one" and three layers qualify.

## Phase 2: Quality Checks

### G1: Methodology
- Train accuracy 99.45% — well above the 92% minimum. Model is well-trained.
- Fit via `torch.linalg.lstsq` on augmented [h_in | 1] — correct formulation for affine fit.
- r computed on held-out data (second 50% of samples, disjoint from fit set) — correct.
- r = ||h_out − Ah_in − b||_F / ||h_out − mean(h_out)||_F — correct normalization (variance-normalized residual).

### G2: Reproducibility
- Single seed (42). Flagged but acceptable for exploratory work. The effect size is large (r=0.05 vs threshold 0.15).
- Hyperparameters fully recorded in JSON artifact.

### G3: Tautology Check
- Is deep-layer linearity trivially expected? Partially: as ReLU kills more neurons (dead fraction increases with depth), the surviving affine map becomes simpler. The dead+always-active fraction at L4 is 43.8%, meaning nearly half the neurons are stable. However, the result is NOT trivially a tautology — linearity depends on training dynamics (which neurons become stable), not architecture alone. A randomly initialized network would not show this pattern.
- **Flagged as partial tautology concern.** The finding is genuine but the strength of the effect is partly explained by neuron death in deep layers.

### G4: Scope and Connection
- Stays within item 20 scope.
- Directly relevant: identifies which layers can be compressed via single affine collapse.
- The confirmed result unlocks items 21 (GPT-2 analog), 27 (persistent homology), 29 (local complexity).

### G5: Counterintuitive Results
Not counterintuitive — deeper layers linearizing is consistent with the "Deep Grokking" literature and the neuron stability statistics.

## Status: PASS

original_outcome: confirmed
audited_outcome: confirmed
guardrail_flags: G2 (single seed), G3 (partial tautology — deep layer linearity partly from neuron death)
reason: Three of four layers have r well below 0.15, confirming that trained deep ReLU layers are near-linear on the data distribution. The effect is genuine and large enough to survive single-seed concerns. The tautology flag is noted but does not invalidate the finding.
recommendations: Test on GPT-2 (item 21) to see if the pattern transfers. Also run items 22/37 which explore multi-region structure in the nonlinear first layer.
