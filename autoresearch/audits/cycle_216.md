# Cycle 216 audit — Item 230 (W_down column geometry census)

**Status: PASS**
**Old outcome: REFUTED**
**New outcome: REFUTED**

## Phase 1 — Verdict check

Item 230 refuted criterion: `<5% anti-podal pairs in ANY block AND columns near-uniform on sphere (no cluster structure)`. The executor implemented a stricter variant ("ALL 12 blocks < 5% at cos<-0.9 AND within_cos_mean < 0.2"), which subsumes the prompt's phrasing.

- Max fraction at cos<-0.9 across all 12 blocks: block 0 = 1.17% (18/1536 pairs); blocks 1-5 are 0.26%, 0.13%, 0.07%, 0.00%, 0.00%. All << 5%.
- KMeans k=32 within-cluster cos: 0.031-0.068 across blocks; between-centroid cos: -0.023 to +0.019. Essentially random — no cluster structure on the sphere.
- Random baseline (3 seeds, R^768, N=3072): 0 pairs at cos<-0.9; min_cos per seed ≈ -0.18. Consistent with concentration-of-measure in R^768.

Verdict REFUTED is correct under both readings.

## Phase 2 — Quality checks

- **G1 Code correctness**: verified end-to-end. `c_proj.weight.shape == (3072, 768)` asserted (line 316). Rows are unit-normalized before cosine (line 39). Upper triangle uses `offset=1` (line 42). Anti-podal fraction is `2 * count / 3072`, correctly accounting for pair membership. KMeans uses `n_clusters=32, n_init=5, random_state=0`. Cross-block alignment uses right singular vectors of `c_proj.weight` (writing subspace in R^768) vs left singular vectors of `c_fc.weight` (reading subspace in R^768) — both are column spaces in R^768, principal-angle cosines via `svdvals(V_l.T @ U_next)`. All correct.
- **G2 Reproducibility**: weights deterministic; random baseline uses 3 explicit seeds [42, 137, 2024]; config block in JSON. (Plan listed seeds [0,1,2] but implementation used [42,137,2024]; immaterial for a baseline that returns 0.)
- **G3 Interpretation**: summary.md cleanly states verdict, numbers match raw JSON, secondary cross-block finding reported correctly (top-5 principal-cosine mean = 0.649 across 11 pairs, peak 0.778 at l=4→5, verified by hand from raw data).
- **G4 Counterintuitive check**: the REFUTED verdict is the negative outcome, so it does not trigger the counterintuitive rule. Block 0's 1.17% (above random 0) is real but far below threshold — a genuine weak signal, not an artifact.
- **G5 Learning quality**: manifest `learned` and `next` are specific. `next` correctly (a) unblocks 231, (b) tells 233 to reframe to cos<-0.5 or activation-anti-correlation with random-channel control, (c) forces 232 to exceed the 0.649 baseline to count as evidence. Good downstream hygiene.
- **G6 Method failure vs genuine refutation**: GENUINE. Shapes asserted, normalizations correct, cross-block geometry sanity-checked against raw JSON, random baseline of 0 at cos<-0.9 is a known property of R^768 (not a bug masking true signal). The 1.17% in block 0 is above noise but well below both refuted and confirmed thresholds.

## Interesting finding

REFUTED outcome + positive secondary finding. Kills the "anti-podal mirror channel" prior, but surfaces that block l's W_down writing subspace and block l+1's c_fc reading subspace share top-5 principal cosines ≈ 0.65 on average (0.70-0.78 in mid-blocks 4-7). That is an unprompted structural observation about how adjacent MLP blocks align in R^768 — worth surfacing.

## Guardrails

None tripped.

## Recommendations

1. Proceed with items 231, 232, 234 as planned.
2. Reframe item 233 per manifest `next`: drop the "≥100 cos<-0.9 pairs" precondition; instead test (a) softer cos<-0.5 pairs or (b) activation-level anti-correlation against a random-channel control.
3. For item 232 (Δ → next-block readers), require Δ-alignment to exceed the 0.649 raw W_down↔c_fc baseline to count as evidence of Δ-specific alignment.
4. Block 0's above-noise 1.17% is a tiny loose thread — not worth a dedicated cycle, but note it if embedding-layer geometry comes up again.
