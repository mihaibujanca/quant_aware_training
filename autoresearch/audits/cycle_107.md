# Audit: Cycle 107 — Output-Space Direct Regression Bypass — GPT-2 FFN

**Item:** 117
**Claimed outcome:** CONFIRMED
**Audited outcome:** INCONCLUSIVE
**Status:** DOWNGRADE

## Phase 1: Verdict Check

Original criteria: "Confirmed if direct rank-r regression matches or beats topk+lstsq at matched param count in >=2/4 blocks, for any r."

The cycle claims blocks 4 and 7 as wins. Block 4 wins only at r=32 (delta=-0.505%), topk wins at all other ranks. Block 7 wins at r=32-256 (delta -0.12% to -0.41%), topk wins at r=384. The verdict logic counts a block as a "direct win" if any single rank has delta<=0. This is mechanically correct per the criteria wording ("for any r"), but the wins are 0.1-0.5% — within noise for this eval setup.

## Phase 2: Quality Checks

### G1: Implementation correctness

**Direct regression: CORRECT.** SVD(X) basis -> rank-r projection -> lstsq fit Y = X@V_r@A^T + b -> nn.Linear with rank-r weight W = A@V_r^T. Mathematically sound rank-constrained least-squares.

**Topk implementation: CORRECT but uses atlas data, not per-seed data.** The topk MLP is built from `atlas.layer_geometries` (probed once before the seed loop) using `layer.hidden_sample` and `layer.output_sample`. This is deterministic across seeds — see G2.

**Param matching: CORRECT.** Direct: 2*768*r + 768. Topk: 2*768*r + r + 768. Difference is r params (<0.07%), negligible.

### G2: Seeds — FLAG: pseudo-replication

**3 seeds [42, 123, 7] but topk PPLs are IDENTICAL across all seeds.** The seed only affects the probe token sampling for the direct method. The topk baseline is computed once from the atlas (built before the seed loop) and produces the exact same model for every seed. This means:

- Direct: N=3 genuine runs with different probe data
- Topk: N=1 (same model evaluated 3 times, yielding identical PPL)

The paired t-test reported in the manifest is comparing 3 draws of the direct method against a constant. It tests whether the direct method's mean differs from a single topk point estimate, but tells us nothing about topk's sensitivity to data sampling. The "3 seeds" claim is misleading — this is effectively an unpaired comparison with N_direct=3 and N_topk=1.

### G3: Statistical significance — FLAG: wins are within noise

The "winning" deltas are tiny:
- Block 4, r=32: -0.505% (0.35 PPL on a base of ~70.7)
- Block 7, r=32: -0.386% (0.28 PPL on a base of ~73.0)
- Block 7, r=64: -0.411% (0.30 PPL)
- Block 7, r=128: -0.296% (0.22 PPL)

The direct method's cross-seed std at block 7 ranges from 0.11 to 0.60 PPL. The block 7 wins at r=128 and r=256 have absolute deltas (0.09-0.28 PPL per seed) that are smaller than the cross-seed std of the direct method itself. At block 7, r=128, individual seed differences are [-0.09, -0.64, +0.08] — one seed actually loses.

With the topk baseline being a single point (no variance estimate), and the direct method showing sub-PPL-point wins with high relative variance, these results cannot be distinguished from noise. A proper comparison would need either (a) the topk method also re-fitted per seed on different probe subsets, or (b) many more seeds to narrow the direct method's confidence interval.

### G4: What does "CONFIRMED" mean here?

Even taking the numbers at face value: the best direct wins are 0.3-0.5% PPL improvement in middle blocks, while early/late blocks show 2-27% degradation. This means direct regression can approximate middle-block FFNs about as well as topk+lstsq (neither is good at compression — both show only ~0.5% PPL degradation from float because these blocks contribute little). The practical value is limited: you wouldn't use direct regression because it works about the same as topk in easy blocks and much worse in hard blocks.

### G5: Insight

The genuine finding is the block-dependent structure: middle blocks (4, 7) have near-linear FFN behavior (direct linear bypass works), while early (1) and late (11) blocks rely on GELU nonlinearity. This is useful interpretability but doesn't require a CONFIRMED verdict on the compression question.

### G6: Baseline correctness

The topk+lstsq baseline uses atlas-probed data for selection and fitting, which is the standard pipeline. The implementation looks correct (topk importance selection, lstsq output projection, GELU activation). No issues with the baseline itself — the problem is that it wasn't re-fitted per seed.

## Verdict

**DOWNGRADE to INCONCLUSIVE.** The wins are 0.1-0.5% in middle blocks, which is within noise given (a) topk has zero variance across seeds (pseudo-replication), (b) direct method's cross-seed std exceeds the win margins at several configs, and (c) only 3 seeds with an effectively unpaired comparison. The criteria say "Inconclusive if results are block-dependent or within 5%" — both conditions apply.

## Recommendations

1. Re-run with topk also re-fitted per seed (sample different probe subsets for atlas per seed) to get a fair paired comparison.
2. Use more seeds (10+) or bootstrap confidence intervals on the PPL comparison.
3. The block-dependent linearity finding is the real value here — consider a focused interpretability item measuring effective linearity of FFN per block.
