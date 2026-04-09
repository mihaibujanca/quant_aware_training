# Cycle 336 Brief

## Item
Item 344: Static rank-2 correction at b4/b5 — bake the LeakyReLU² leak as a fixed correction, use ReLU² everywhere

## Context
- Item 344 proposes a rank-2 SVD correction at blocks 4 and 5 to approximate the lrelu2 activation effect using ReLU² everywhere.
- Hypothesis: "static correction recovers ≥50% of the lrelu² advantage over relu² without retraining."
- Dependency 343 (tested as 343b in cycle 335): selective lrelu2 at b4/b5 only.

## Dependency Result
Cycle 335 (item 343b) verdict: **REFUTED-distributed-mechanism**.
- global-lrelu2 BPB: 1.3931
- selective (lrelu2 at b4/b5): 1.5085
- global-relu2: 1.5740
- Selective recovered only 0.0655 of the 0.1809 gap = **36.2%**.
- The lrelu2 benefit is distributed across all 13 blocks, not localized to b4/b5.

## P2 Check: PLAN INVALID
Item 344's rank-2 SVD correction at b4/b5 is a *strict approximation* of the actual lrelu2 activation at those blocks. If the actual activation change at b4/b5 only recovers 36% of the full advantage, a rank-2 approximation of that change will recover ≤36% — well below the 50% threshold required for CONFIRMED. The item's premise is invalidated by its dependency.

## Key Files
- `autoresearch/cross_project/parameter_golf.md`
- `autoresearch/results/cycle_335_summary.md`
- `autoresearch/results/cycle_335_pgolf_selective_lrelu2.json`

## Verdict
PLAN INVALID — dependency 343 REFUTED, mechanism is distributed, 2-block correction cannot reach ≥50% recovery.
