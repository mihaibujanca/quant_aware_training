# Audit — Cycle 030

## Item
18 — Learned compressor generalization on held-out blocks

## Phase 1: Verdict Check

Claimed: **inconclusive** — generalization works within narrow depth bands (1/4 held-out blocks) but not broadly.

Criteria: "Inconclusive if: generalization works only within narrow depth bands or only within one checkpoint."

Evidence: block 10 (deep) gets good compression from the shared model, but blocks 1, 5, 8 don't. The shared model underfits overall. Per-block distillation is much better.

**Verdict correct.** 1/4 blocks is not "majority" for confirmed, and the shared model does work partially, so not refuted.

## Phase 2: Quality Checks

### G1: Methodology
- Shared U, V factorization + per-block bias — reasonable architecture choice.
- 8 train / 4 test split — sufficient.
- 200 steps of training — may be undertrained. But the fundamental issue is architectural (blocks compute different functions), not training budget.

### G3: Tautology Check
Not tautological. The finding that blocks don't share reusable structure is empirical.

## Status: PASS

original_outcome: inconclusive
audited_outcome: inconclusive
guardrail_flags: G1 (200 training steps may be insufficient; but architectural mismatch is the root cause)
reason: GPT-2 FFN blocks compute qualitatively different functions — no shared low-rank structure to learn. Per-block compression (with or without SGD) is the viable path.
