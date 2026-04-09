# Cycle 371 Brief: Residual Stream Representation Geometry (Item 379)

## Context
**Previous cycle (370)**: Selective compression (skip b10+b11) → PPL 150 ± 0.65. Refuted the high-error-blocks hypothesis. Composition barrier is **count-driven, not identity-driven**: 3-block PPL~47, 6-block PPL~65, 10-block PPL~150, 12-block PPL~186.

**Working recipe**: topk+lstsq + MLP adapter. Single-block beats float (46.22 vs 53.89). 3-block ceiling at 47.43. All attempts to escape the 6-block barrier (adaptive k, KD, selective skip, decorrelation) have failed.

## Item 379: Core Hypothesis
We know WHERE error explodes (blocks 10–11, 77% of FFN error) and that attention does NOT amplify error (c366). But we don't know WHY composition collapses.

**Measurement targets**: Effective rank of residual stream activations, inter-token cosine similarity, isotropy score (variance in top-5 PCs) at block boundaries for 0, 3, 6, and 12 compressed blocks.

## Goal
Detect geometric representation collapse that correlates with PPL explosion onset:
- **Confirmed if**: eff_rank drop >30% or isotropy drop >50% separates working (3-block) from failing (6-block) configs
- **Refuted if**: geometry indistinguishable between 3-block and 6-block
- **Inconclusive if**: metrics shift but don't cleanly separate working from failing

## Key Insight
Cycle 370 proved composition failure is structural: even optimal block subset (uniform, including b10/b11) fails at 6 blocks, while 3 blocks work. This points to cumulative geometric degradation — not local error magnitude, but how error compounds through representation space itself.
