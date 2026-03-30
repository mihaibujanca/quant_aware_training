# Cycle 124 Review — Attention Entropy Shift under FFN Compression

Item 141. Reviewer: design-review agent.

## R1: Premise validity — OK

FFN compression changes intermediate representations that become keys/values for
downstream attention layers. Expecting measurable entropy shifts is reasonable.
The correlation with cycle 104 cascade amplification fractions is a natural
follow-up. The compressed blocks [1,4,7,11] are sparse (4 of 12), so entropy
shifts in non-compressed blocks would reflect indirect propagation through the
residual stream — the plan should explicitly distinguish direct vs indirect when
reporting.

## R2: Measurement soundness — MINOR FIX NEEDED

**Spearman threshold too lenient.** The plan correlates 11 data points
(transitions 0-10). With n=11, |Spearman| > 0.3 has a two-tailed p-value around
0.37 — nowhere near significant. The confirmation criterion must include a
significance gate: require p < 0.10 (or better, p < 0.05) alongside |rho| > 0.3.
The plan already says it will report p-value, but the verdict logic in step 6
ignores it. Fix: add "AND p < 0.10" to the CONFIRMED correlation condition.

**Null baseline is fine.** Float-float entropy difference is zero by construction.
The plan correctly identifies this.

## R3: Capacity and statistical adequacy — OK

3 seeds, 2048 tokens each, ~16 query positions per sequence. For detecting mean
entropy shifts of 0.02 nat resolution, this is adequate. Cross-seed std check
(< 0.05 nats) is a good guard.

## R4: Method-architecture alignment — MINOR FIX NEEDED

**Hook target is wrong.** The plan says "hook on `attn.c_attn`" but `c_attn` is
the QKV linear projection — its output is the concatenated Q, K, V vectors, not
the post-softmax attention weights. Hooking `c_attn` would require manually
splitting Q/K/V, computing scaled dot-product, masking, and applying softmax —
reimplementing HuggingFace internals.

**Simpler approach:** Use `model.config.output_attentions = True` and read
`outputs.attentions[layer]` which returns post-softmax attention weights with
shape `[batch, n_heads, seq, seq]`. Verified this works on GPT-2 (standard and
custom models with modified FFNs, since attention modules are untouched by FFN
compression). The fallback in the plan (manual QKV split) is unnecessary if
`output_attentions=True` is the primary path.

**Action:** Replace the hook-based approach with `output_attentions=True`. Remove
the fallback or keep it only as a defensive check.

## R5: Known pitfall scan — OK

- **MP7 (Post-LayerNorm):** Attention entropy is computed on softmax outputs, not
  on hidden states directly. LayerNorm affects the Q/K magnitudes but not the
  relative angular structure that determines attention patterns. Not a violation.
- **MP1 (Threshold above trivial):** 0.1 nats is ~2% of max entropy range
  [0, ln(128)] = [0, 4.85]. Reasonable detection threshold, well above noise.
- **MP6 (Verdict misclassification):** The plan has a clean three-way verdict
  (confirmed/refuted/inconclusive) with distinct conditions. OK.
- No MP3/MP5/MP8 issues.

## Summary

Two minor fixes needed, neither blocking:
1. Add significance gate (p < 0.10) to the Spearman correlation verdict criterion.
2. Use `output_attentions=True` instead of forward hooks on `c_attn`.

**Verdict: ACCEPT with minor fixes.**
