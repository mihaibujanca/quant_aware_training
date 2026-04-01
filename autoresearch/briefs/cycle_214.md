# Cycle 214 Brief: Analytical W_out Correction (Item 224)

## Result: REFUTED

Analytical lstsq on compressed-model activations with float-model targets:
- Init PPL: 61.746
- After analytical correction: 61.767 (no improvement, −0.021 PPL)
- Adam correction @1000 steps: 48.111 (+13.7 PPL better)

**Why analytical correction fails**: The compressed model's block inputs have a
different distribution than the float model's block inputs (cascade corruption from
earlier compressed blocks). lstsq fits W_out to map {compressed_context_act → float_output},
which is a distribution mismatch. The original lstsq init already uses float context
activations → float outputs; using compressed context activations doesn't help.

**Why Adam succeeds**: The CE loss gradient captures the GLOBAL downstream effect of
each block's output on the final prediction. Adam updates W_out to minimize full-model
cross-entropy, effectively adapting the block to work within the cascade context.
Analytical correction is a LOCAL operation that cannot capture this global signal.

The cascade correction problem is fundamentally non-local: it requires gradient descent
on the full model CE objective.
