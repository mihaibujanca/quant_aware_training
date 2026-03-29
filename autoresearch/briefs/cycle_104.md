# Cycle 104 Brief — Item 115

## Item
Cascade error source: residual-stream vs nonlinearity amplification — GPT-2 FFN

## Context
- **Binding constraint**: c≈0.41/block cascade is bottleneck on stacking (ratio-/order-independent, cycles 089, 093)
- **GELU anatomy confirmed** (cycle 103): 94.6% activations in bump zone, 84-88% disagreement from negative leak
- **Question**: Where does 41% amplification originate — GELU nonlinearity or attention redistribution?
- **Hypothesis space**: (a) GELU sensitivity in bump zone pushes errors into higher-gradient regions; (b) attention heads redistribute error across sequence

## Dependencies
None. Independent analysis, uses existing quantized models from prior cycles.

## Key Files
- Quantized GPT-2 models: saved from cycles 089–103
- Activation analysis code: `notebooks/canonical_error_correction.py`, hooks for forward traces
- Prior work: cycle 103 (bump zone, derivative analysis)

## Constraints
- k=1 compressed blocks only (isolate individual pair dynamics)
- Measure: (1) FFN input error δ_L at next block, (2) FFN output error with/without attention, (3) GELU derivative at perturbed/unperturbed z
- Blocks [1, 7, 11]: represent early/mid/late positions (11 transitions, need ≥8/11 clear dominant source)
- Success: one source ≥70% across ≥8 transitions → actionable; 40-60% split → inconclusive

## Handoff
Write brief decomposition notebook: (1) load k=1 compressed model, (2) capture error δ through residual at L+1 input, (3) compute GELU′(z) at perturbed z vs baseline, (4) toggle attention between L and L+1, (5) tabulate cascade fractions per block pair. Target: 25-line decision.
