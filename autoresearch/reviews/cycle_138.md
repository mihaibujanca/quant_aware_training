# Cycle 138 Review — AE k=4 Stacking (Item 152)

Item 152. Reviewer: design-review agent.

## R1: Premise validity — OK

Cycle 137 CONFIRMED AE's architectural advantage at single-block (3-3.4% at
matched params). Testing whether local advantage compounds under stacking is
the direct next question. The cascade-dominance null hypothesis is well-motivated
by cycle 104 (70%+ attention-driven cascade).

## R2: Measurement soundness — OK

Three conditions (topk, AE, AE+LoRA) on the same blocks, same corpus, same
polish budget. Per-block AE training before stacking correctly isolates the
"local quality -> stacking" question. Joint polish on all 4 blocks simultaneously
matches the working recipe protocol. Pre- and post-polish measurements at each
stage prevent the cycle 133 failure mode (polish masking real differences).

## R3: Capacity and statistical adequacy — MINOR NOTE

Single-seed initial sweep is a reasonable scoping decision given ~5h CPU wall
time. However: the CONFIRMED threshold (PPL < 67, i.e., >4% improvement) is
tight enough that a single seed could cross it by noise. The escalation protocol
(rerun with 2 more seeds if threshold crossed) handles this correctly. No change
needed, but the implementer should understand: a single-seed "confirmed" is
provisional until 3-seed replication.

## R4: Method-architecture alignment — OK

The plan correctly trains AE per-block independently (matching cycle 137 protocol)
then stacks. This is the right design: joint AE training across 4 blocks would
confound "AE architecture advantage" with "joint training advantage."

One concern from cycle 137 review carries forward: d_bn=64 dead neuron risk.
The plan should log dead neuron fractions per block, especially since blocks 1
and 11 were not tested in cycle 137 (which only ran blocks 4 and 7). If block 1
or 11 shows >30% dead neurons, the per-block AE may be weaker there, affecting
the stacking comparison unfairly.

## R5: Known pitfall scan — OK

- LoRA on attention projections (condition 3): the plan assumes LoRA infra exists
  for the modified model (AE-swapped blocks). LoRA hooks onto attention, not FFN,
  so AE swap should not interfere. But verify that the LoRA forward hooks remain
  intact after AE swap. The plan's fallback (defer condition 3) is appropriate.
- Param accounting: AE uses ~996K for 4 blocks vs topk ~1.57M. AE has FEWER
  params. This means an AE advantage could be partially attributed to
  regularization from the bottleneck, not just better reconstruction. This is
  fine — the question is practical (does AE give better PPL?), not mechanistic.
- No MP violations. No underdetermined systems at experiment level.

## Summary

No blocking issues. Two advisory notes:
1. Log dead neuron fractions for all 4 blocks (especially untested blocks 1, 11).
2. Single-seed CONFIRMED is provisional — the escalation protocol is correct
   and must be followed before declaring a final verdict.

**Verdict: ACCEPT.**
