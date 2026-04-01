# Cycle 181 Review — Full-model ReLU + bias swap (Item 191)

Item 191. Reviewer: self-review agent.

## R1: Premise validity — OK

Cycle 173 confirmed per-block: C3 (replace negative regime with conditional mean) recovers within +2% PPL in 12/12 blocks tested individually. The per-block bias μ_neg_l = W_down @ E[GELU(z) ⊙ 1[z≤0]] is a valid first-order correction for the GELU→ReLU difference. The open question is composition: do 12 simultaneous swaps cascade?

Cycle 166 found +87% for full-model ReLU+bias, but that used b_leak in intermediate space added to the activation (ReLU(z) + b_leak). Here the bias is added to the block output (c_proj bias). These are mathematically equivalent (W_down @ (ReLU(z) + b_leak) = W_down @ ReLU(z) + W_down @ b_leak), so we should expect the same +87% result unless the μ computation differs.

**Potential issue:** If μ_neg_l is computed on the float model's activations but applied to a model where earlier blocks already have ReLU, the bias estimate may be stale for later blocks. The plan computes all μ_neg_l from the float model (all blocks GELU), then swaps all 12. This is the standard approach but worth noting.

## R2: Measurement soundness — OK

200 eval sequences, same as baseline. Incremental accumulation curve (blocks 0..k) will reveal cascade pattern. The comparison to cycle 166 (+87%) provides a built-in consistency check.

## R3: Capacity and statistical adequacy — OK

2048 tokens for μ_neg_l estimation is sufficient (cycle 173 used 200×1024=200K tokens for stats; 2048 is specified in item text). Single seed is acceptable for a binary confirmed/refuted test with wide thresholds (+10% / +50%).

## R4: Method-architecture alignment — NOTE

The μ_neg_l is computed from float model activations. When all 12 blocks are swapped, block l's actual input distribution shifts because blocks 0..l-1 are already modified. This distribution shift means μ_neg_l underestimates the true correction needed for later blocks. The incremental curve will reveal this.

## R5: Known pitfall scan — OK

- No GELU/ReLU confusion: we explicitly compute GELU(z) ⊙ 1[z≤0] on the float model and add the bias to ReLU output.
- No architecture mismatch: we modify the existing model in-place.
- Baseline sanity check included.

## Summary

Clean experiment. The most likely outcome is REFUTED or INCONCLUSIVE given cycle 166's +87% result, but item 191's thresholds are generous (+50% for refuted). The incremental curve adds diagnostic value beyond the binary verdict.

**Verdict: APPROVE.**
