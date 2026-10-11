# Item
237 (P0): Cycle 218 specificity control — Δ̃_l alignment with non-adjacent c_fc(j).
For each l ∈ {0..10}, compute top-20 right SVs of centred Δ̃_l and measure top-5 principal cosine sums against c_fc(j) for ALL j > l. Plot heatmap (l, j). Compare row's max-at-j=l+1 against row's mean over non-adjacent j > l+1.

# Context
- Cycle 218 (CONFIRMED): centred Δ̃ top-20 right SVs vs c_fc(l+1) top-20 left SVs → cos_sum 4.42-7.07 (+17 to +33σ above random 2.72±0.10) in 11/11 cross-block pairs.
- Audit (G3) flagged: missing specificity control. If Δ̃_l ALSO aligns with c_fc(j) for j ≠ l+1, the alignment is generic (Δ̃ leaks into a universal reading basis), not specifically inter-block.
- This is the audit follow-up control test.

# Dependencies
- Item 232 / cycle 218 (done) — provides centred Δ̃ infrastructure.

# Key Files
- experiments/delta_centred_alignment_cycle218.py — has the centred Δ̃ pipeline. Reuse.

# Constraints
- Use the SAME 200-sequence calibration batch as cycle 218.
- For each l = 0..10:
  - Compute centred Δ̃_l.
  - SVD → top-20 right SVs (Vh[:20]).
  - For ALL j ∈ {l+1, l+2, ..., 11}:
    - U_j = top-20 left SVs of c_fc(j).weight (R^768).
    - cos_sum_top5 = svdvals(Vh[:20] @ U_j)[:5].sum()
    - cos_sum_top20 = svdvals(Vh[:20] @ U_j).sum()
- Random baseline: 10 seeds Gaussian QR-orthonormal 20×768 vs each U_j.
- Build heatmap: (l, j) cos_sum, (l, j) significance vs random.
- Verdict criteria from agenda:
  - Confirmed if median (l, l+1) cos_sum exceeds the per-row mean over non-adjacent j by ≥ 1.0 AND non-adjacent (l, j) cos_sums sit within mean+3σ of the random baseline in ≥ 7/11 rows
  - Refuted if non-adjacent (l, j) cos_sums are within 0.3 of (l, l+1) on average across rows
  - Inconclusive if peak-vs-mean gap is 0.3-1.0

# Handoff
Plan agent: design via reusing cycle 218. New script `experiments/delta_specificity_cycle227.py`. Heavy reuse of cycle 218. Output JSON with the (l, j) heatmap data + summary markdown.
