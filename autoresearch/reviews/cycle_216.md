# Cycle 216 Design Review — Item 230

## Verdict: APPROVE
## Round: 1

## Summary
Plan is sound. Conv1D layout is correctly handled, SVD directions are correct
(right singular vectors of c_proj and left singular vectors of c_fc both live in
R^768), threshold and baseline methodology are reasonable, and runtime budget
is realistic. No blocking issues.

## R1 — Premise
PASS. The hypothesis is concrete and grounded in the GELU-bump lineage: W_down
column directions in R^768 determine where the FFN can write, and anti-podal
pairs would effectively double channel capacity. First weight-geometry census
in this lineage (cycle 195 was Jacobian rank, 201 was input/output SVD — not
column geometry). No duplication.

## R2 — Measurement
PASS with commentary.

- **Conv1D layout**: Plan correctly treats `c_proj.weight` as shape (3072, 768)
  and slices rows as W_down columns in R^768. Shape assertion at step 4.
- **Anti-podal threshold**: cos<−0.9 for 3072 unit vectors in R^768 under a
  uniform-on-sphere null is essentially zero (spherical cap volume ~ 1e-70).
  Plan's explicit random baseline with 3 seeds will confirm this numerically.
  20% of columns having an anti-podal twin is therefore strong signal, not a
  near-baseline trigger. The ≥10× fold-over-random requirement from the brief
  is tracked in `aggregate.twin_fold_over_random`.
- **Pair-to-column normalization**: `pct_columns_with_twin = 100 * 2 * n_pairs
  / n_cols` is correct if each column has at most one strong anti-pode (the
  generic case); capped at 100 handles the rare multi-twin case. Acceptable.
- **Cos matrix memory**: 3072×3072 fp32 = 37.7 MB, fits trivially.
- **SVD**: `torch.linalg.svd(W_down_cols, full_matrices=False)` on (3072, 768)
  returns S shape (768,) — valid.
- **KMeans**: `n_init=5` with sklearn's default kmeans++ init is standard. Plan
  handles empty/singleton clusters (`if pts.shape[0] < 2: continue`). Fine.
- **Cluster quality metrics**: "within > 0.5 AND between < 0.2" is a
  conjunction, which prevents the two failure modes (tight but not separated,
  or separated but not tight) from confirming alone. The conjunctive criterion
  is the right design.

## R3 — Capacity / Baseline
PASS. No learning, no capacity risk. Random baseline is computed with 3 seeds
for the anti-podal test AND for KMeans cluster quality. Fold-over-random is
recorded explicitly.

## R4 — Method-Architecture Alignment
PASS. This was the main risk point; plan handles it correctly.

**SVD direction — critical verification:**
- Block l "writing subspace" in R^768: the plan uses `Vh[:20, :].T`, i.e. the
  top-20 RIGHT singular vectors of `c_proj.weight ∈ R^(3072, 768)`. Right
  singular vectors of an (m, n) matrix live in R^n — here R^768. Correct.
  Geometrically: this is the span in R^768 of the rows of c_proj.weight =
  column space of c_proj.weight^T = image of W_down: R^3072 → R^768. Correct.
- Block l+1 "reading subspace" in R^768: the plan uses `U_next[:, :20]`, i.e.
  the top-20 LEFT singular vectors of `c_fc.weight ∈ R^(768, 3072)`. Left
  singular vectors of an (m, n) matrix live in R^m — here R^768. Correct.
  Geometrically: this is the column space of c_fc.weight = row space of
  W_up (when W_up is written 3072×768) = the subspace of R^768 that W_up
  actually reads from. Correct.
- Principal angles: `M = V_l.T @ U_next[:, :20]` → (20, 20), then
  `svdvals(M)` gives cosines of principal angles in [0, 1]. Standard formula
  for subspace alignment; both inputs have orthonormal columns (from SVD), so
  this is correct.

Note: the brief text at line 27-28 said "top right singular vectors of W_up_{l+1}
= c_fc.weight" — this phrasing is slightly loose (right SVs of c_fc.weight live
in R^3072, not R^768). The plan correctly resolves this to LEFT SVs of c_fc,
which is the geometrically correct subspace in R^768. Plan overrides brief
phrasing correctly.

## R5 — Known Pitfalls
- **MP1 (threshold above trivial baseline)**: The 20% threshold is WAY above
  null (spherical cap at cos<-0.9 in R^768 is ~0). Explicit random baseline
  confirms. PASS.
- **MP2 (activation regime mismatch)**: N/A — weights only, no activations.
- **MP3 (underdetermined fitting)**: N/A — no fitting.
- **MP4 (phenomenon-domain mismatch)**: N/A — measured directly on GPT-2.
- **MP5 (insufficient capacity)**: N/A — no training.
- **MP6 (verdict misclassification)**: Conjunctive confirm/refute criteria plus
  INCONCLUSIVE default protect against this. A block with structure-but-no-
  twin-pairs lands in INCONCLUSIVE, not REFUTED. PASS.
- **MP7 (post-LayerNorm)**: N/A — operating on raw W_down weights, not on
  activations. Weight geometry is not distorted by the downstream LN before
  attention. PASS.
- **MP8 (representation vs functional rank)**: This is a function-geometry
  measurement (SVD of the W_down linear map), not a representation-cloud
  measurement. Plan does the right thing. PASS.

## Minor observations (non-blocking)
- `cumsum < 0.95` for rank_95 should be `cumsum / total < 0.95` (already in
  plan, step 5g: `(S.cumsum(0) / S.sum() < 0.95).sum() + 1`). Correct.
- `torch.triu_indices(3072, 3072, offset=1)` returns (2, ~4.7M) — acceptable.
- KMeans input is `cols_norm` (unit-normalized), so Euclidean KMeans is
  effectively spherical KMeans up to a monotone transform. Reasonable for
  cluster geometry on the sphere.
- Cluster sizes list per block is stored (good for post-hoc audit if a block
  falls into INCONCLUSIVE).
- Fallback to torch-only KMeans if sklearn missing is sensible.

## Actionable feedback
None. Plan is ready to execute as written.
