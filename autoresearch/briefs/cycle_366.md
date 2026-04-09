# Item 375

**Residual stream error propagation monitor** — trace per-token error through full GPT-2 forward pass (R5)

# Context

Composition barrier is superlinear despite orthogonal per-block errors (c362). Single-block PPL ~46, 3-block ~48, 6-block ~65, 12-block 186. Need to identify WHERE error growth rate transitions from linear to superlinear across 12 blocks.

# Dependencies

None. Independent item.

# Key Files

- `experiments/gpt2_*_cycle*.py` — reference structure for GPT-2 instrumentation
- Wikitext-2 eval harness (standard, 200 sequences)
- Model: GPT-2 with 6-block topk+lstsq+adapter recipe (k=512, h=64 adapters, per c359)

# Constraints

- Float baseline: PPL 53.89
- 6-block compressed: PPL ~65
- Measure ||h_compressed - h_float|| at each of 12 block outputs (not just FFN error, but cumulative after attention)
- Threshold: identify block range where error growth rate **doubles** (inflection point)
- 200 sequences sufficient for statistical stability

# Handoff

Instrument the forward pass to log residual stream norm after each block attention+FFN. Compare float vs 6-block-compressed pass token-by-token. Look for where error magnitude transitions from growing linearly to exponentially. This identifies which blocks drive the composition barrier.
