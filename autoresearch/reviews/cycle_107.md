verdict: APPROVE
round: 2

## R1: Issue resolution

**I1 (rank-constrained lstsq): FIXED.** Step 3 now projects X onto its top-r right singular vectors before solving lstsq, producing the optimal rank-r linear map in the data-dependent least-squares sense. This is the correct analog of what topk+lstsq does on its selected hidden units.

**A1 (reconstruction error): ADDRESSED.** Both steps 3 and 4 record ||Y - Y_hat||_F / ||Y||_F. The plot layout (step 6) includes a dedicated second row for reconstruction error vs param count. If direct regression achieves lower reconstruction error but worse PPL, that isolates the GELU non-linearity's contribution — good diagnostic.

**A2 (seed clarification): ADDRESSED.** Step 1 explicitly states seeds control probe token sampling and that lstsq is deterministic given data.

## R2: Param matching

Step 4 claims 1536k = 1536r (i.e., 2×768×k = 2×768×r → k=r). Direct regression: A (768×r) + V_x[:,:r]^T (r×768) = 2×768r. Topk: up (768→k) + down (k→768) = 2×768k. Matched when k=r. Both methods add one bias (768). Correct.

## R3: Known pitfalls

- **MP3:** At rank r, effective features = r. Worst case r=384: ratio 4096/384 ≈ 10.7:1. Clear.
- **MP6:** Verdict logic uses PPL deltas with explicit thresholds (0% and 5%), not uninformative metrics. Not at risk.
- **MP1-2, MP4-5, MP7-8:** Not applicable (no thresholds on activations, no cross-domain transfer, no capacity concern, no post-LN measurements, no intrinsic dimension estimation).

## R4: Remaining observations

No blocking or advisory issues. The plan is clean and self-contained.
