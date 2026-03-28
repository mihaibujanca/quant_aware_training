status=CORRECTED
old_outcome=confirmed
new_outcome=confirmed (with caveat: lstsq refit dominates, SVD basis not special vs random at k/n=0.67)
reason=SVD+lstsq beats topk+lstsq by >=3% in 2/4 blocks (blocks 4 and 11), meeting the literal confirmation criterion. However, random+lstsq nearly matches SVD+lstsq (gap 0.15-0.71%), meaning the Jacobian SVD basis provides negligible advantage over random orthogonal projection at k=512/768. The real finding is "any rotation+refit >> axis-aligned topk", not "Jacobian directions are special." Param count reporting is misleading (393K vs 788K) but actual storage is matched (~787K each when U_k is counted).
guardrail_flags=G3 (interpretation: CONFIRMED is technically correct but misleading — the mechanism is rotation-invariance of lstsq at high k/n, not Jacobian spectral structure), G4 (counterintuitive: random ≈ SVD needs explanation — at k/n=0.67, random subspaces have high overlap with any target subspace)
tsv_path=autoresearch/logs/experiments.tsv
recommendations=Rerun at k=128 and k=256 where SVD vs random should diverge if Jacobian directions matter. The current k=512/768=67% retention is too generous to distinguish basis quality.
interesting=true
interesting_reason=random+lstsq matching SVD+lstsq reveals that at high retention ratios (k/n>0.5), lstsq refit makes basis choice irrelevant — the bottleneck is axis-alignment of topk, not which rotation you use. This has practical implications: skip expensive Jacobian SVD computation and use random projection + lstsq instead.
