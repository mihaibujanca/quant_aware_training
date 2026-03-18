# Audit — Cycle 022

## Item
40 — Attention head geometry census — GPT-2

## Phase 1: Verdict Check

Claimed: **inconclusive** — C1 met (89.6% heads low-rank), C2 partially failed (K=5 ok at +2.7%, K=10 at +14.6% exceeds 5%).

Criteria:
- Confirmed: ≥20% heads eff_rank < 0.3×head_dim AND zeroing 10 ≤5% ppl. C1 met (89.6% >> 20%), C2 not met (K=10 = +14.6%).
- Refuted: all heads high eff_rank AND zeroing 5 > 15%. Not met (most heads are low-rank, K=5 = +2.7%).

**Verdict correct.** Strong geometric signal but compounding removal degrades fast.

## Phase 2: Quality Checks

### G1: Methodology
- Effective rank = nuclear_norm / spectral_norm — standard measure.
- CKA between heads within layer — appropriate for redundancy.
- Head zeroing by nullifying c_proj rows — correct implementation.
- Prunability score combining rank, variation, redundancy — reasonable composite.

### G2: Reproducibility
Deterministic. Acceptable.

### G3: Tautology Check
Low effective rank for attention heads is partly expected: attention matrices are softmax outputs (rows sum to 1), which constrains the rank. However, the effective rank values (1-7 for deep layers) are much lower than the maximum possible under the softmax constraint, so this is a genuine training-dependent finding.

### G4: Scope
Extends beyond FFN-only scope (program says "FFN-only compression first"). But the agenda explicitly included this item. Provides useful context for the full compression picture.

## Status: PASS

original_outcome: inconclusive
audited_outcome: inconclusive
guardrail_flags: G3 (softmax constrains attention rank — partly expected but magnitude is genuine)
reason: Attention heads are geometrically low-rank (89.6% < 0.3×d_head) and K=5 can be safely pruned (+2.7% ppl). But compounding is superadditive, same pattern as FFN blocks in cycle 018. This is the most consistent finding across all experiments: individual components are redundant but collective removal is costly.
recommendations: The compounding pattern across both FFN blocks and attention heads suggests that sequential, interleaved compression (remove one component, fine-tune, repeat) is necessary. Pure no-SGD compression has a hard ceiling imposed by compounding effects.
