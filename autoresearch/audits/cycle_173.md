# Cycle 173 — Audit

**Item:** 184. Unified per-block regime ablation table — GPT-2 FFN
**Verdict:** CONFIRMED

## G1: Baseline sanity
- Baseline PPL: 22.70 (within [15, 65]). PASS.

## G2: Consistency with prior cycles
- C1 block 0: +233% PPL. Cycle 102 had block 0 at +2664% (full ReLU swap, which also changes positive regime). Our C1 only zeros z<0 output, keeping gelu(z) for z>0. The difference is expected: ReLU(z) = z for z>0, vs gelu(z) which is slightly different. PASS.
- C2 block 0: +970k% PPL. Cycle 164 had block 0 at +970k%. Exact match. PASS.
- C1 blocks 1-11: range +4-34%. Cycle 102 had +8-56% (full ReLU is more aggressive). Consistent direction. PASS.

## G3: Verdict correctness
- Criterion: C3 within +2% in >=8/12 blocks.
- Result: C3 within +2% in 12/12 blocks. Max delta = +1.4% (block 4).
- Margin is overwhelming (12/12, not borderline 8/12). PASS.

## G4: Artifact completeness
- JSON with full 12x6 table, norms, interaction check: EXISTS.
- PNG heatmap: EXISTS.
- Manifest: EXISTS.
- Agenda updated (NEXT→PAST): YES.
- experiments.tsv appended: YES.

## G5: Code correctness
- Hook correctly reads input[0] for pre-activation z, output for gelu(z). PASS.
- C3 replaces negative-regime channels with conditional mean E[gelu(z)|z<0]. PASS.
- Stats collected in float64 to avoid accumulation errors. PASS.
- Each condition hooks only the target block; all others remain float. PASS.

## G6: Key findings
1. Negative regime is functionally a constant bias: replacing token-dependent negative with its mean costs at most +1.4% PPL in any block.
2. Positive regime is critical: killing it (C2) is catastrophic, especially block 0 (+970k%) and block 11 (+89%).
3. Block 0 is uniquely sensitive: only block where killing negative (C1) causes >100% damage (+233%), and constant output (C5) causes +5798%.
4. Interaction gap is always negative: independent kills (C1+C2-C0) sum to much more than constant replacement (C5), meaning regimes partially compensate each other.
5. C4 (replace positive with mean) causes +2.5% to +11.1%, showing token-dependent positive signal matters but is not catastrophic per-block.

## Status: PASS
