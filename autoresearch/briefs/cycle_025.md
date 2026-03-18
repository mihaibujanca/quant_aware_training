# Brief — Cycle 025

## Item
33 — GELU → PWL approximation fidelity on GPT-2

## Motivation
Cycle 016 (item 32, toy GELU MLP): 3-piece drops 35pp accuracy (39% relative), 10-piece drops 2.6%.
GELU is not well-approximated by 3 pieces on the toy, but 10 pieces is excellent.
Question: does the LLM scale change this? Transformers are over-parameterized and residual connections
may absorb local approximation error per block. If so, 3-piece is usable and partition-based theory
(SplineCam, RTG, MASO) applies to GELU networks without retraining.

## Hypothesis
At LLM scale, residual connections + layer norm absorb per-block GELU approximation error.
Predicted outcome: 3-piece causes moderate (~10–20%) ppl increase, not catastrophic (>20%).
If all 12 blocks are replaced simultaneously, damage compounds multiplicatively.

## Method
- Load GPT-2 + Shakespeare probe (same setup as cycles 005, 018, 024)
- Baseline perplexity (~122 from prior experiments)
- Replace ALL GELUs with K-piece PWL (K = 3, 5, 10, 20) over [-5, 5]
  - Outside range: 0 for x < -5, x for x > 5 (GELU asymptotes)
- Block-by-block sweep: replace only block i's GELU, measure ppl increase
- Monkey-patch `model.transformer.h[i].mlp.act` for each block

## Criteria
- Confirmed: 3-piece causes < 5% relative ppl increase
- Refuted: ppl increases > 20%
- Inconclusive: 5–20% increase

## Key context
- GPT-2 GELU activations live in `model.transformer.h[i].mlp.act` (nn.GELU module)
- 17% of toy activations outside [-3, 3]; using [-5, 5] for LLM (pre-act values larger)
