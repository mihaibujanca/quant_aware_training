# Cycle 143 Review — Eigenstructure-Aware Channel Selection (Item 155)

Item 155. Reviewer: design-review agent.

## R1: Premise validity — OK

Cycle 129 established that FFN activation covariance is non-trivial (7-8x isotropy
increase). Cycle 141 showed raw importance gives a small, inconsistent advantage
over topk. The question "does decorrelating before scoring help?" is the natural
next step. If the covariance structure encodes redundancy, whitening should expose
it. If the covariance is irrelevant to channel selection (e.g., because lstsq
already absorbs covariance effects in the refit), this is still a clean negative
result. Sound premise.

## R2: Measurement soundness — OK

Three-way comparison (topk vs importance vs whitened) at matched k=512 eliminates
the param confound. Pre-polish / post-polish dual reporting follows the cycle 133
lesson. The Jaccard overlap diagnostic is a valuable addition — if whitened and
topk select the same channels, the experiment is testing a distinction without a
practical difference, and this is caught early.

## R3: Capacity and statistical adequacy — OK

3 seeds x 4 blocks x 3 methods = 36 triplets. Sufficient for paired comparison.
The 2% threshold on relative PPL is appropriate given cycle 141 showed 0.2-2% for
raw importance. If whitened can't clear 2%, it's not operationally useful.

## R4: Method-architecture alignment — OK with one note

**Lstsq refit absorbs linear covariance effects.** The lstsq step solves
min ||H_kept @ W_out - Y||. If two channels are perfectly correlated, lstsq
already assigns near-zero weight to the redundant one. Whitening changes WHICH
channels are selected, but lstsq then optimally recombines whatever channels are
kept. This means whitening only helps if the set of k=512 channels it selects
spans a materially different subspace of the output than topk's selection. The
plan's Jaccard diagnostic partially addresses this; the implementer should also
log the lstsq reconstruction MSE for all three methods (before PPL eval) to see
whether different channel sets give different local approximation quality.

## R5: Abort conditions — OK

The condition-number check (if kappa ~ 1, whitening is a no-op) and Jaccard > 0.95
early-inconclusive are good abort conditions. The eigenvalue regularization fallback
is sensible.

## Verdict: APPROVE

Clean incremental test. The main risk is that lstsq refit washes out selection
differences (cycle 133 pattern), which is why pre-polish numbers are essential.
The plan correctly includes them.
