# Cycle 227 Audit — Item 237 (Δ̃_l specificity vs non-adjacent c_fc(j))

**Status: PASS**

## Verdict check
- Criteria: median (l, l+1) cos_sum gap ≥ 1.0 AND ≥ 7/11 rows with non-adjacent (l, j) within mean+3σ random → CONFIRMED; non-adjacent within 0.3 of (l, l+1) → REFUTED; gap 0.3–1.0 → INCONCLUSIVE.
- Reported: median_specificity = **+0.791** ∈ [0.3, 1.0); rows with all non-adjacent quiet = **0/10**; non-adjacent above 3σ = **55/55**.
- 0.791 < 1.0 and 0 < 7 → INCONCLUSIVE per literal criterion. Verdict matches.

## Cycle 218 sanity reproduction
Adjacent top-5 cos_sum mean = 2.96866 (cycle 227) vs 2.9687 (cycle 218), Δ = −0.00004. Same model, same data loader, same hooks, same centred Δ̃ construction (lines 96–125). The 11 adjacent (l, l+1) entries reproduce cycle 218's signal exactly. (Note: cycle 218's cos_sums of 4.42–7.07 cited in audit-G3 are top-20; the 2.9687 is the top-5 mean. Both heatmaps are stored in the JSON.)

## Implementation review
- c_fc reader basis = top-20 left singular vectors of `model.transformer.h[j].mlp.c_fc.weight` ∈ R^(768×3072) → U[:, :20] ∈ R^(768×20). Correct: c_fc reads from a 768-dim residual subspace, so left SVs are the right reader basis.
- Δ̃_l writer basis = top-20 right SVs of centred Δ̃_l = (GELU − ReLU)(c_fc input) @ W_proj, ∈ R^(N×768). Correct.
- Cosine sum: `Vh_top20 @ U_fc[j]` → svdvals → top-5 / top-20 sum. Standard principal-cosine reduction.
- Random baseline: 10 seeds × Gaussian QR-orthonormal R^(20×768), per j. Top5 baselines 1.21–1.27 ± 0.03–0.06 — tight enough to give 3σ thresholds 1.34–1.43, well below the smallest non-adjacent observed (1.52 at l=0,j=10).

## Per-row pattern
Specificity gap (max@l+1 − mean over j>l+1) is positive in every l ∈ {0..9}: range 0.40–1.00. l=4 hits the highest peak (3.32 at j=5) but also has the highest non-adjacent floor (j=11 still 1.69). Adjacency preference is real but small relative to broad-band write into all c_fc(j>l).

## Scientific reading (G3 — interpretation)
Cycle 218's CONFIRMED finding "Δ̃_l writes into c_fc(l+1)'s reading subspace at +17–33σ above random" survives, but the implication "specifically targets the next block" does not. Cycle 227's heatmap shows Δ̃_l also writes into every c_fc(j), j > l+1, at well above random — the universal-c_fc-reading-basis interpretation cannot be ruled out from this measurement. The +0.79 PPL adjacency preference is positive but modest relative to a baseline that itself sits 0.3–1.5 above the random floor. Plausible mechanism: Δ̃_l is high rank (cycle 218 reported rank₉₅ ≈ 552 / 768) so its column space necessarily overlaps with any 20-dim reader. The narrative correction is genuine, not a method bug (G6 clear).

## Bookkeeping
manifest, exec_result, JSON, summary all consistent (item=237, cycle=227, outcome=inconclusive, median_specificity=0.791, n_rows_quiet=0/10, sanity OK). agenda.md PAST/Cycle 227 carries the full numbers and the universal-reading-basis caveat.

## Recommendations
- **Update cycle 218 interpretation** in literature_map / program / future briefs: "Δ̃ writes into the universal downstream c_fc reading basis with mild adjacency preference (+0.79 top-5 cos_sum gap)" — not "specifically into block l+1".
- **Direct functional test** to disambiguate "preferential write at l+1 with broad-band leak" vs "universal reading basis": swap c_fc(l+1) for c_fc(j>l+1) at block l write time, measure PPL effect. Item 237's claim cannot be settled by SVD geometry alone — it needs a causal intervention.
- **Tighter reader subspace** (rank_50 or rank_10) might surface stronger specificity if the broad-band leak is concentrated in low-σ tails of c_fc. Cheap follow-up.
- **Inverse-distance weighting** would test smooth decay with j−l (visible in the heatmap top-20: 4.85 → 3.40 from l=0 row).
