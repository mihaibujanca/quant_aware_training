# Cycle 341 Audit

## Item 350: Output-covariance spectrum census on GPT-2

## Methodology
- Reuses cycle 278 output-covariance data (200 wikitext-2 val seqs, 128 tokens, seed 42, mlp.c_proj hooks, C = Y^T Y / N, eigh, exp(entropy) eff_rank) — identical measurement, no need to re-run GPU inference.
- Computes top-10/top-13 energy fractions from stored top-20 eigenvalues (verified: top-1 and top-5 match stored values to < 1e-3).
- Compares against cycle 338 weight SVD per block.

## Sanity checks
- [x] Top-1 and top-5 computed from eigenvalues match stored cycle 278 topk_energy values
- [x] Block count: all 12 blocks present
- [x] Concentration amplification ratios are plausible (3.2x-14.0x)

## Verdict assessment
Item 350 says CONFIRMED if >= 3 blocks have top-5 output-covariance energy >= 50%.
Result: 6 blocks (1,2,3,4,10,11) have top-5 >= 50%. CONFIRMED.

## Red flags
- None. Data is deterministic reuse of cycle 278 with additional computed columns.

## Key finding
Weight SVD is flat across all blocks (5-12% top-5), but output covariance is highly block-dependent (19.7%-99.6% top-5). The data-dependent concentration is 3-14x the weight-intrinsic concentration. This confirms that LASER viability is a data-dependent property, not a weight-intrinsic one.
