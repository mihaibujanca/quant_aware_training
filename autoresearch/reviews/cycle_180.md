# Cycle 180 Review — FLAP-style Bias Compensation (Item 190)

Item 190. Reviewer: self-review agent.

## R1: Premise validity — OK

The FLAP formula is straightforward: removed channels contribute E[W_down[:,j] * h_j] to the output on average. Absorbing this as a constant bias is a first-order correction. The comparison is fair — same channels selected by topk, only the output reconstruction method differs (analytic bias vs lstsq).

## R2: Measurement soundness — OK

Three conditions share identical topk selection. The key difference: condition A uses original W_down columns for kept channels (no data fitting), condition B fits W_down columns via lstsq. Condition C tests whether polish can close the gap. Per-block MSE comparison will show where the gap originates.

## R3: Capacity and statistical adequacy — OK

3 seeds, 200 eval sequences. Verdict thresholds (3 PPL for bias-only, 2 PPL for bias+polish) are wide enough that seed variance (~1-2 PPL) won't cause false positives.

## R4: Method-architecture alignment — NOTE

The analytic bias uses mean(hidden_sample[:,j]) for removed channels. This mean is computed on calibration data, so it is NOT fully calibration-free. The key advantage over lstsq is that it requires only mean statistics (O(n_tokens) per channel) rather than solving a (512 × 768) linear system. If the agenda item intended fully calibration-free, this is a misalignment — but the item text says "on calibration data", so this is consistent.

A subtlety: condition A uses original W_down[:, selected] as c_proj weight. This is exact for the kept channels individually, but doesn't account for the interaction between kept channels' activations and the removed channels' bias. lstsq jointly optimizes all kept-channel weights to minimize reconstruction error including these interactions. So we expect condition A to be worse than B, and the question is how much.

## R5: Known pitfall scan — OK

- No architecture mismatch (same CompressedMLP structure).
- Dead channels: handled by topk's dead_mask exclusion.
- No risk of GELU/ReLU confusion — we use hidden_sample which is already post-GELU.

## Summary

No blocking issues. The experiment cleanly tests whether analytic bias compensation (cheap) can substitute for lstsq reconstruction (expensive).

**Verdict: ACCEPT.**
