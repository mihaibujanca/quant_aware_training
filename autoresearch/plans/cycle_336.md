# Cycle 336 Plan

## Status: PLAN INVALID

**P2 — dependency 343 REFUTED, mechanism is distributed, 2-block correction cannot recover ≥50%.**

Item 344 proposes a rank-2 SVD correction at blocks 4 and 5 to replicate the lrelu2 advantage using ReLU² everywhere, with a ≥50% recovery threshold for CONFIRMED.

Cycle 335 tested the dependency (item 343b): keeping the *actual* lrelu2 activation at b4/b5 (with relu2 elsewhere) recovered only 36.2% of the full lrelu2 advantage (selective BPB 1.5085 vs global-lrelu2 1.3931, gap 0.0655 out of 0.1809 total). A rank-2 approximation of the activation difference is strictly weaker than the actual activation change, so the correction at 2 blocks is upper-bounded at ~36% recovery — below the 50% CONFIRMED threshold.

## Action
Move item 344 from NEXT/P1 to PAST as INCONCLUSIVE (plan invalid, untested due to dependency refutation).
