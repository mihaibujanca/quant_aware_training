# Cycle 141 Review — Graduated Importance Pruning (Item 153)

Item 153. Reviewer: design-review agent.

## R1: Premise validity — OK

Cycle 128 showed binary pruning is vacuous (0 strict-removable). Graduated
importance is the natural next step: if no channel is fully dead, some are
presumably less important than others. The importance metric (weight norm *
activation RMS) is standard and captures both the channel's potential (W_down
column norm) and its actual utilization (GELU activation RMS). Sound premise.

## R2: Measurement soundness — OK with key safeguard

The matched-k design eliminates the param confound: both methods keep the same
number of channels, differing only in selection criterion. The pre-polish /
post-polish dual measurement directly addresses the cycle 133 lesson (polish
erases differences). This is the critical methodological addition.

One concern: topk is per-input adaptive (different channels survive per token),
while importance pruning is static (same channels dropped for all tokens). This
is not a flaw — it is the core question — but the plan correctly identifies it
in the hypothesis section.

## R3: Capacity and statistical adequacy — OK

3 seeds x 4 blocks x 3 pruning levels gives 36 paired comparisons per checkpoint
(pre/post polish). More than sufficient to detect a 1% PPL difference given
typical seed variance (~0.1-0.5 PPL). The 2048 probe tokens for importance
ranking match the census methodology from cycle 128.

## R4: Method-architecture alignment — MINOR NOTE

The importance metric is global (computed once over 2048 tokens) while topk is
local (per-input). This asymmetry is intentional and is exactly what is being
tested, but the implementer should note: importance pruning + lstsq refit is
essentially a static subnetwork, while topk + lstsq is a data-dependent
subnetwork. If importance loses, it could be because static selection is
fundamentally worse, not because the importance metric is bad. A future item
could test data-dependent importance (per-token importance ranking), but that
is out of scope here.

## R5: Abort conditions — OK

Early abort if pre-polish numbers match is efficient. The 3-min-per-block
timeout is reasonable.

## Verdict: APPROVE

Sound design. The pre-polish / post-polish dual reporting is essential and
correctly specified. The matched-k control eliminates confounds. Low risk,
fast execution, clean answer either way.
