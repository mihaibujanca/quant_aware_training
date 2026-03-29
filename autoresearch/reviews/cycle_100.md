verdict: APPROVE
round: 2

## Round 1 issue resolution

| ID | Status | Notes |
|----|--------|-------|
| I1 (blocking) | FIXED | SVD now performed on mean Jacobian J_bar (768×768). Shapes explicit: U_full[:, :r] → (768,r), V_full[:, :r] → (768,r), S(x) = diag(U^T J(x) V) → length-r. Mathematically correct. |
| I2 (blocking) | FIXED | Refutation gate added: median relative reconstruction error > 0.5 at r=128 → REFUTED. Present in both hypothesis section and Phase 2 steps. |
| I3 (blocking) | FIXED | Predictor changed from MLP to ridge regression (sklearn Ridge, CV over alpha). 49K params on 1536 samples is still underdetermined by MP3's 2x rule, but ridge with CV is the standard remedy — regularization prevents interpolation, CV selects shrinkage. Acceptable. |
| I4 (minor) | FIXED | 3 seeds (different 512-token eval slices) specified in Phase 4. |
| A1 (advisory) | FIXED | Random U,V (orthogonal) + oracle S now listed as explicit Phase 4 comparison (item c). |
| A2 (advisory) | FIXED | Total params formula now includes predictor: 2×768×r + r×768 + r + r. Example: ~148K at r=64. |

## Known pitfall re-scan

- **MP3**: Mitigated by ridge regression with CV regularization. The ratio is still low (0.03x), but ridge is specifically designed for this regime. Pass.
- **MP7, MP8**: Still passing (unchanged from round 1).
- **MP1, MP2, MP4, MP5, MP6**: Not applicable or passing (unchanged).

No new issues identified.

## Minor observations (non-blocking)

- The fallback "column-stacked SVD" description says "U from projection" without specifying the projection formula. Since this is a fallback and the primary mean-Jacobian approach is sound, this is not blocking.
- Ridge regression eval on only 512 tokens may have high variance for MSE reporting. The PPL evaluation (3 seeds) is the real test, so this is acceptable.
