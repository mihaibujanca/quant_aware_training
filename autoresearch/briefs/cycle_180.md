# Cycle 180 — Brief

**Item:** 190 — FLAP-style bias compensation for topk pruning — GPT-2 FFN k=4
**Type:** empirical
**Depends on:** none (analytic formula from FLAP, applied to existing topk selection)

## Context

Our baseline topk+lstsq+polish achieves ~70.7 PPL at k=4 (512 channels), blocks [1,4,7,11].
The lstsq step requires calibration data and is the most expensive part of compression after probing.

FLAP (Fluctuation-Aware Pruning) proposes an analytic alternative: when removing channel j,
its expected output contribution E[W_down[:,j] * GELU(z_j)] can be precomputed from calibration
data and absorbed as a constant bias correction to the output. This avoids solving any linear
system — just an empirical mean per removed channel, summed into a bias vector.

## What this cycle measures

1. **Condition A — analytic bias only**: topk selection (same 512 channels), no lstsq. For kept
   channels, use original W_down columns directly. For removed channels, compute analytic bias
   Db = sum over removed j of E[W_down[:,j] * GELU(z_j)] and add to c_proj bias.
2. **Condition B — topk + lstsq** (control): standard baseline.
3. **Condition C — analytic bias + polish**: condition A with 500-step Adam polish.

## Criteria

- **Confirmed:** bias-only PPL <= 73 (within 3 of lstsq), OR bias+polish <= 68.7 (2 better than lstsq)
- **Refuted:** bias-only > 85 AND bias+polish within +/-1 of lstsq-only
- **Inconclusive:** bias-only 73-85 with no polish improvement
