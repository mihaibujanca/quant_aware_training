# Cycle 228 Audit — Item 238 (gelu_noneg fair rerun with per-channel μ correction)

**Status: PASS**

## Verdict check
- Criteria: corrected_noneg_ppl ≤ 49.57 → CONFIRMED; ≥ 100 → REFUTED; in (49.57, 100) → INCONCLUSIVE.
- Reported: corrected_noneg_ppl = **61.664**, baseline = 45.064, uncorrected (cycle 220) = 2093.296.
- 49.57 < 61.664 < 100 → INCONCLUSIVE per literal criterion. Verdict matches.

## Implementation review
- μ collected via forward hook on `c_fc` input (LN2 output, line 96–98). Manual recompute: `preact = x_l @ c_fc.weight + c_fc.bias`; `mu_l = (gelu(preact) - relu(gelu(preact))).mean(dim=0)`. Sign convention is `GELU − ReLU(GELU)` ≤ 0 elementwise — by construction since clamping removes the negative tail only.
- μ sanity confirmed in JSON: every block has `n_positive_entries = 0`. Per-block mean(μ) range −0.0701 (block 1) to −0.0950 (block 9), grand mean −0.0874, std 0.014–0.033. Roughly 0.5× cycle 184's reference −0.17.
- `GeluNonegCorrected` instantiated **distinctly per block** (line 168–173, with `assert len(ids_check) == 12`). Critical: shared instance would broadcast a single μ everywhere.
- Patch correct location: `model.transformer.h[l].mlp.act` (post-c_fc, pre-c_proj). Forward = `relu(GELU(z)) + μ` matches the math.
- Restore-and-re-eval: post_restore_baseline_ppl = 45.0637 == baseline_ppl = 45.0637 (line 186 assert at 1e-4). Patching is clean.
- Eval is deterministic (no_grad, fixed seed 42, single eval pass). No seed-jitter band needed.

## Scientific reading (G3 — interpretation)
First-moment correction recovers 2031.6 PPL of the cycle 220 catastrophe — a ~34× shrink in relative degradation (4545% → 36.84%) and ~99.2% of the absolute gap. The residual +16.6 PPL gap is higher-order beyond the constant per-channel mean shift: GELU's negative-zone curvature carries token-conditional second moments and cross-channel covariance that a per-channel scalar bias cannot represent. Cycle 228 lands in the same family as cycle 188 c3_mean (+20.3%) and cycle 190 hybrid (+27.7%); strictly better than cycle 184 ReLU+bias (+87.7%). Not counterintuitive (G4 clear) — consistent with the cascade activation-correction family.

## Bookkeeping
manifest, exec_result, JSON, summary all consistent (item=238, cycle=228, outcome=inconclusive, corrected=61.664, baseline=45.064, μ ≤ 0 verified). agenda.md PAST/Cycle 228 carries the result. The "necessary but not sufficient" framing is in the next-step note. Genuine inconclusive (G6 clear).

## Recommendations
- **Second-moment correction**: precompute per-channel scale s_l matching `var(GELU(z)) ≈ var(s_l · relu(GELU(z)) + μ_l)` on calib. If residual is variance, this closes it; if it's token-conditional covariance, it won't.
- **Cycle 218-style centred Δ̃ projection**: project the gelu→noneg residual into the next block's c_fc reading subspace (rank-20 or rank-50 SVD), absorb as a per-block low-rank affine. Direction is approaching the cycle 188/190 family.
- **Close the gelu_noneg lineage** if neither (1) nor (2) recovers >half the residual 16.6 PPL. The corrigible activation-substitution direction has a known floor around +20–37% for this trainable surface.
