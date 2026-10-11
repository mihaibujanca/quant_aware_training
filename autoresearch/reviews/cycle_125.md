# Cycle 125 Review — Correction Placement Sweep (Item 142)

Item 142. Reviewer: design-review agent.

## R1: Premise validity — OK

Cycle 123 showed V-dominant early layers and QK-dominant late layers in the
attention cascade. Testing whether routing corrections to the error source
(attention subcomponents) outperforms post-hoc residual correction is a natural
next step. The five placements span a reasonable space: output-side (P1),
input-side (P2), attention-internal (P3, P4), and dual residual (P5).

## R2: Measurement soundness — OK

Null baseline is clear: uncorrected PPL = 95.95 (seed 42). P1 reproduction
check against cycle 115 (69.53 +/- 2) is a strong guard against pipeline
regressions. The sanity check that all placements must beat uncorrected PPL
catches broken hooks. Param count verification (49,152 per placement) ensures
fair comparison. No issues.

## R3: Capacity and statistical adequacy — OK with caveat

Seed 42 only for the initial sweep is acceptable for a ranking experiment. The
plan correctly gates multi-seed escalation on a 5% relative threshold. However:
the plan should state explicitly that a single-seed ranking is only for
SCREENING purposes and that the cycle verdict cannot be CONFIRMED from seed 42
alone. If escalation is triggered and 3-seed results flip the ranking, the
verdict must reflect the 3-seed data, not the initial sweep. This is implicit
in the plan but should be explicit in step 8.

## R4: Method-architecture alignment — MINOR FIX NEEDED

**P3/P4 hooks on c_attn: in-place modification risk.** The plan says to "add
U_q @ V_q @ x to Q segment" of the c_attn output. GPT-2's forward does
`self.c_attn(hidden_states).split(self.split_size, dim=2)` — the `.split()`
returns views into the same underlying tensor. If the hook modifies the output
in-place (e.g., `output[:,:,:768] += correction`), this silently corrupts the
original tensor and breaks autograd. The implementation MUST clone before
modifying:

```python
def hook(module, input, output):
    out = output.clone()  # or torch.cat modified segments
    out[:,:,:768] += U_q @ (V_q @ input[0])
    out[:,:,768:1536] += U_k @ (V_k @ input[0])
    return out
```

Or better: construct the correction tensor and return `output + correction`
(where `correction` is zero except in Q/K segments). Addition creates a new
tensor, avoiding the in-place problem entirely.

**P4 O correction input:** The plan says the O correction's input is "attention
output." When hooking `c_proj` with `register_forward_hook`, `input[0]` is the
pre-projection attention output (768-dim) and `output` is the projected result
(768-dim). The correction `output + U_o @ V_o @ input[0]` mixes two different
representation spaces (post-projection output, pre-projection input). This is
valid — it's equivalent to a residual connection around c_proj — but document
this design choice explicitly in the code so future analysis (item 144) doesn't
misinterpret the correction's geometric meaning.

## R5: Known pitfall scan — OK

- **MP5 (Insufficient capacity):** r=4 per sub-projection in P3/P4 is the
  minimum viable rank. At d=768, rank 4 captures only 0.5% of the space. If
  P3/P4 underperform, the conclusion should be "P3/P4 at r=4 underperform P1 at
  r=8" NOT "attention-internal correction is worse." Item 143's rank sweep
  addresses this correctly — just make sure the cycle 125 verdict language
  doesn't overclaim if P1 wins.
- **MP7 (Post-LayerNorm):** P3's input x = ln_1 output. This is correct: c_attn
  also takes ln_1 output, so the correction operates in the same space as the
  original projection. Not a violation.
- **MP3 (Underdetermined):** 200 seqs x 128 tokens = 25,600 training positions,
  49,152 params. Ratio is ~0.5, which is borderline. However, each position
  provides a 768-dim gradient signal and the loss is over the full vocabulary, so
  effective information per sample is much higher than 1. Not a practical issue
  at 2000 Adam steps, but worth monitoring if training loss plateaus at a high
  value (would suggest underfitting, not overfitting, given the loss surface).
- No MP1, MP2, MP4, MP6, MP8 issues.

## Summary

One minor fix, not blocking:
1. P3/P4 c_attn hooks must avoid in-place modification of the output tensor.
   Use `output + correction_tensor` (addition, not indexing assignment) to
   guarantee autograd correctness.

One advisory note (no code change needed):
2. If P1 wins the sweep, the verdict should note that P3/P4 were tested at
   half the effective rank (r=4 vs r=8). Defer the definitive placement
   comparison to item 143's rank-controlled sweep.

**Verdict: ACCEPT with minor fix.**
