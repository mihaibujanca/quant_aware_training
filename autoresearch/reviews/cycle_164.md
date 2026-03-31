# Cycle 164 Review — Positive-channel ablation — GPT-2 FFN (Item 174)

Item 174. Reviewer: design-review agent. Round 1.

## R1: Premise validity — PASS
Clean complement of cycle 102 (which killed negative regime via GELU→ReLU,
found +8320% PPL). Testing whether killing positive regime is equally
catastrophic is well-motivated and has no prior cycle coverage. Sound premise.

## R2: Measurement soundness — PASS (advisory)
PPL is a direct task metric. Condition A (zero) is clean: zero output where
z > 0, keep GELU(z) where z ≤ 0. Verdict logic uses Condition A only.

**[A1]** Condition B (mean replacement): the current formula replaces positive
channels with `gelu(z).mean(dim=1)`, which averages over ALL channels
(positive and negative). For a clean "token-mean replacement" of only the
positive regime, it should average only over positions where z > 0. This
doesn't affect the verdict (which uses Condition A), but makes Condition B
harder to interpret. Consider:
`pos_mean = (gelu(z) * (z > 0)).sum(dim=-1, keepdim=True) / (z > 0).sum(dim=-1, keepdim=True).clamp(min=1)`

## R3: Capacity and statistical adequacy — PASS
200 sequences is consistent with cycle 102. Hooks are deterministic (no
learned parameters, no dropout), so single-seed eval with fixed data order
is adequate. Per-block sweep (12 evaluations) provides sufficient granularity.

## R4: Method-architecture alignment — PASS
Plan correctly identifies `NewGELUActivation` and uses z > 0 as the
threshold (not GELU(z) > 0), avoiding the sign ambiguity near z ≈ 0.
Hook intercepts c_fc output (pre-activation), applies conditional masking,
then lets GELU act on surviving channels. Pseudocode is correct.

## R5: Known pitfall scan — no triggers
- MP2 (activation regime): handled — plan explicitly uses pre-activation
  sign (P4 confound), not GELU output sign.
- MP6 (verdict misclassification): PPL is direct task metric with clear
  thresholds (>100% = confirmed, <20% = refuted). Pass.
- MP1, MP3-5, MP7-8: not applicable to this ablation design.

## Summary
Well-designed ablation that cleanly complements cycle 102. No blocking
issues. One advisory on Condition B's mean computation (does not affect
verdict).

**Verdict: APPROVE**
