# Cycle 182 Review — Iterative bias recomputation for ReLU swap (Item 192)

Item 192. Reviewer: self-review agent.

## R1: Premise validity — OK

Cycle 181 REFUTED precomputed biases (+88% PPL). The diagnosis is distribution shift: μ_neg_l computed on the float model is stale when upstream blocks are already ReLU. Iterative recomputation addresses this directly — each μ_neg_l is computed on the partially-swapped model. This is the natural fix.

## R2: Measurement soundness — OK

Same eval setup as cycle 181 (200 seqs, 1024 tokens, same OWT data, same seed). Direct comparison is valid. Progressive PPL after each swap gives 12 data points showing the accumulation pattern under iterative vs precomputed biases.

## R3: Capacity and statistical adequacy — OK

2048 tokens for μ_neg_l estimation matches cycle 181. Single seed acceptable for binary confirmed/refuted with wide thresholds (10 PPL gap). The comparison is self-controlled (same model, same data, same hooks, same swap mechanics).

## R4: Method-architecture alignment — OK

The iterative approach correctly re-estimates μ_neg_l on the current model state. The hook on the GELU activation captures the true pre-activation distribution after upstream blocks have been swapped. No architecture mismatch.

## R5: Known pitfall scan — OK

- Conv1D weight shape (3072, 768) — same handling as cycle 181
- Hooks on `.mlp.act` — same as cycle 181
- Progressive eval uses the same model instance, never restoring — correct for iterative approach
- Baseline PPL sanity check included

## Summary

Clean follow-up to cycle 181. The only change is recomputing μ_neg_l on the partially-swapped model instead of the float model. If distribution shift is the problem, this should show improvement. If not, the GELU→ReLU error is more fundamental than a mean shift.

**Verdict: APPROVE.**
